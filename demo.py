import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

# TOY DATA
demoTitle1 = "Donald Trump and Joe Biden set to compete in upcoming election"
demoComment1 = "I hate this."
demoComment2 = "I also hate this."
demoComment3 = "I love this."

demoTitle2 = "Filibuster in the house lead by senator McConnell"
demoComment4 = "I hate this."
demoComment5 = "I also love this."
demoComment6 = "I love this."

# Perform NER on post titles
def NER(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_results = nlp(text)

    return ner_results

# BERT searches for the smallest unit of a name that it recognizes and so often splits up names into multiple parts
# (eg. B, ##iden)
# So this function combines all parts of names into one name string
# The way it works necessitates that all parts of a name form one consecutive string
# (eg. DonaldTrump)
# This does lead to errors where if only the name Trump appears in one post title and Donald Trump appears in another,
# they would be categorized as two separate names.
# This is, however, unavoidable because to merge names on the basis of similarity might also merge names of two different people
# that share a first or last name.
def extractNames(textNER):
    names = {}
    removeHashtags = str.maketrans('', '', '#')
    for id,post in textNER.items():
        names[id] = []
        nameParts = []
        nameInds = []
        for unit in post['post']['title'][1]:
            if "PER" in unit['entity'] :
                nameParts.append(unit['word'])
                nameInds.append(unit['index'])

        # print(id, nameParts)

        if len(nameParts) > 1:
            # Pad out end of lists so the else statement triggers at the end no matter what.
            nameInds = nameInds + [nameInds[-1]+1] + [nameInds[-1]+3]
            nameParts = nameParts + ["NULL"] + ["NULL"]
            subNameParts = []

            for ind, name in list(zip(nameInds, nameParts))[:1]:
                # print("0", id, ind, name)

                if nameInds[1] - nameInds[0] > 1:
                    nameStr = name.translate(removeHashtags)
                    names[id].append(nameStr)

            i=1
            for ind, name in list(zip(nameInds, nameParts))[1:]:
                # print("1+", id, i, ind, name)
                if nameInds[i] - nameInds[i-1] == 1:
                    # print(name)
                    if nameParts[i-1] not in subNameParts:
                        subNameParts.append(nameParts[i-1])
                    if name not in subNameParts:
                        subNameParts.append(name)
                else:
                    if len(subNameParts) > 0:
                        if "NULL" in subNameParts:
                            subNameParts.remove("NULL")
                        nameStr = ''.join(subNameParts).translate(removeHashtags)
                        names[id].append(nameStr)
                        subNameParts = []
                i+=1
        elif len(nameParts) == 1:
            # print("==1", id, nameParts[0])
            names[id].append(nameParts[0])

    return names

# Perform sentiment analysis on comments within posts
# returns whether the overall sentiment was positive or negative
# as well as the total "score" which is the sum of all confidence metrics
# (negative if sentiment is negative and positive if sentiment is positive)
def sentimentAnalysis(post):
    pipe = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

    inputs = []
    for ind, comment in post["comments"].items():
        inputs.append(comment)

    outputs = pipe(inputs)

    total = 0
    for out in outputs:
        if out["label"] == "POSITIVE":
            total += out["score"]
        elif out["label"] == "NEGATIVE":
            total -= out["score"]

    if total > 0:
        return "POSITIVE", abs(total)
    elif total < 0:
        return "NEGATIVE", abs(total)
    else:
        return "AMBIGUOUS", 0


# generate fake post data
print("Generating Fake Post Data")
postTextNER = {}
postTextNER["id1"] = {}
postTextNER["id1"]["post"] = {}
postTextNER["id1"]["post"]["title"] = []
postTextNER["id1"]["post"]["title"].append(demoTitle1)
postTextNER["id1"]["post"]["title"].append(NER(demoTitle1))
postTextNER["id1"]["comments"] = {1: demoComment1, 2: demoComment2, 3: demoComment3}

postTextNER["id2"] = {}
postTextNER["id2"]["post"] = {}
postTextNER["id2"]["post"]["title"] = []
postTextNER["id2"]["post"]["title"].append(demoTitle2)
postTextNER["id2"]["post"]["title"].append(NER(demoTitle2))
postTextNER["id2"]["comments"] = {1: demoComment4, 2: demoComment5, 3: demoComment6}

# Extract names from post titles and remove posts without names in their title
print("Extracting Names")
names_and_ids = extractNames(postTextNER)
for id, names in names_and_ids.items():
    if len(names) > 0:
        postTextNER[id]["names"] = names
    else:
        postTextNER.pop(id)

# Performing Sentiment Analysis
print("Performing Sentiment Analysis")
for post in postTextNER.values():
    try:
        post["sentiment"] = sentimentAnalysis(post)
    except:
        post["sentiment"] = ("ERROR", 0)

# Combine all sentiments into a dict with names as keys and a list of all sentiment scores as values
# Count the occurrences of each name and store that in a dict
print("Collecting Data into Dicts")
nameSentiments = {}
nameOccurrence = {}

for post in postTextNER.values():
    for name in post["names"]:
        nameSentiments[name] = []
        nameOccurrence[name] = 0

for post in postTextNER.values():
    for name in nameSentiments.keys():
        if name in post["names"]:
            nameSentiments[name].append(post["sentiment"])
            nameOccurrence[name] += 1

# Add all sentiment scores together and store in a dict
# Average total sentiment scores by the number of posts and store that in a dict
nameTotalSentiment = {}
nameAvgSentimentPerPost = {}

for name, sentiments in nameSentiments.items():

    total = 0
    for sentiment in sentiments:
        if sentiment[0] == "POSITIVE":
            total += sentiment[1]
        elif sentiment[0] == "NEGATIVE":
            total -= sentiment[1]

    nameTotalSentiment[name] = total
    nameAvgSentimentPerPost[name] = total / nameOccurrence[name]

# Visualizations
print("Displaying Visualizations")

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

data = pd.Series(nameOccurrence)
ax1.pie(data.values, labels=data.index, autopct='%.0f%%', radius=1)
ax1.set_title("% of posts each name appears in", y=1.3)

data = pd.Series(nameTotalSentiment)
sns.barplot(x=data.values, y=data.index, hue=data.index, ax=ax2)
ax2.set_title("Total Sentiment")

plt.tight_layout()
plt.show()
