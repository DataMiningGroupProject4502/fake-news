import pandas as pd

#Get CSV Dataframes from the CSV files
file1 = pd.read_csv('classifying the fake news –.csv', encoding = 'utf-8-sig')
file2 = pd.read_csv('Detecting Fake News with Python.csv', encoding = 'utf-8-sig')
file3 = pd.read_csv('Kaggle Samrat Sinha.csv', encoding = 'utf-8-sig')
file4 = pd.read_csv('WELFake_Dataset.csv', encoding = 'utf-8-sig')

#combine all the dataframes into one
combinedFrame = pd.concat([file1, file2, file3, file4])
combinedFrame = combinedFrame[combinedFrame['text'].notna()]


#convert the combined frame into a list
lst = combinedFrame.values.tolist()
for i in lst:
    i[0] = i[0].encode('utf-8') #use the list to encode the items that neeed encoding


#turns the list back into a dataframe for further manipulation
combinedFrame = pd.DataFrame(lst, columns=['text', 'label'])
combinedFrame['text'] = combinedFrame['text'].astype('|S')
combinedFrame['text'] = combinedFrame['text'].str.decode('utf-8')
combinedFrame['lst'] = combinedFrame.text.str.split(' ').tolist()

#converts dataframe back to list one more time
lst = combinedFrame.values.tolist()

#establishing counter variables
realCounter = 0
realTotalWords = 0
fakeCounter = 0
fakeTotalWords = 0

#main loop
for i in lst:
    if i[1] == 0: #if real, add 1 to real counter and add length to the total words
        realCounter += 1
        realTotalWords += len(i[2])
    elif i[1] == 1: #if fake, add 1 to fake counter and add length to the total words
        fakeCounter += 1
        fakeTotalWords += len(i[2])
    else:
        pass

#get the average count
realAvg = realTotalWords // realCounter
fakeAvg = fakeTotalWords // fakeCounter


print("The average word count for real articles is: " + str(realAvg))
print("The average word count for fake articles is: " + str(fakeAvg))


