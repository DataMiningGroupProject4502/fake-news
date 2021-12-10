import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

#convert files into data grames
file1 = pd.read_csv('classifying the fake news –.csv', encoding = 'utf-8-sig')
file2 = pd.read_csv('Detecting Fake News with Python.csv', encoding = 'utf-8-sig')
file3 = pd.read_csv('Kaggle Samrat Sinha.csv', encoding = 'utf-8-sig')
file4 = pd.read_csv('WELFake_Dataset.csv', encoding = 'utf-8-sig')

#combine data frames
combinedFrame = pd.concat([file1, file2, file3, file4])
combinedFrame = combinedFrame[combinedFrame['text'].notna()]

#convert dataframes to lists and encode
lst = combinedFrame.values.tolist()
for i in lst:
    i[0] = i[0].encode('utf-8')

#convert back to data frame
combinedFrame = pd.DataFrame(lst, columns=['text', 'label'])
combinedFrame['text'] = combinedFrame['text'].astype('|S')
combinedFrame['text'] = combinedFrame['text'].str.decode('utf-8')
combinedFrame['lst'] = combinedFrame.text.str.split(' ').tolist()

#back to list
lst = combinedFrame.values.tolist()

#establish lists
realLst = []
fakeLst = []

#main loop to get all words from lists
for i in lst:
    if i[1] == 0:
        realLst = realLst + i[2]
    elif i[1] == 1:
        fakeLst = fakeLst + i[2]
    else:
        pass

#join the lists into a long string
realStr = (" ").join(realLst)
fakeStr = (" ").join(fakeLst)

#generate a word cloud from the text
realCloud = WordCloud(width = 1000, height = 500).generate(realStr)
fakeCloud = WordCloud(width = 1000, height = 500).generate(fakeStr)

#show and save the word clouds
plt.figure(figsize=(15,8))
plt.imshow(realCloud)
plt.axis("off")
plt.savefig("realCloud"+".png", bbox_inches='tight')
plt.show()
plt.figure(figsize=(15,8))
plt.imshow(fakeCloud)
plt.axis("off")
plt.savefig("fakeCloud"+".png", bbox_inches='tight')
plt.show()

plt.close()