# Cricket Data Analysis

Tool to scrape commentary data from cricinfo to perform data analytics.

This Readme will have a blog element and a technical element. For the blog element, I will attempt to explain my mindset during each commit I make, delving into my decision making and also what I have learned from completing the specific tasks.

## Blog

### Match Data

The match data class is an extension to the Match class that is present in th espncricinfo python lubrary. Initially I intended to use the match class as is, however I found that the unmodified class did not satisy my paticular requirements, namely that it did not provide me a way of downloading the full match commentary json object. Therfore I needed to extend the match class myself, and build a function that would grab the full match commentary. Of course, as is with all programming tasks, this was far easier said than done. The first issue comes when we follow the link for the full match commentary. It can be seen that the commentary is paginated in 5 over chuncks, though through observation of the link pattern, it is easy enough to get another section of the innings' commentary.
```
Base URL for detailed commentary
https://hs-consumer-api.espncricinfo.com/v1/pages/match/comments?lang=en&seriesId={seriesid}&matchId={matchid}&inningNumber={inning}&commentType=ALL&sortDirection=DESC

Start from specific over
&fromInningOver={overnumber}
```
From the above, we can see that if we append onto the main link, the specifier for which over to display, we can get the json represented commentary for the whole match.

A helpful parameter in the json commentary object is the 'nextInningOver' which tells the next paginated over, or null, if there are no futher overs in the innings. We can use this parameter to find out the total length of the inning, and then loop through all the paginated sections of the commentary until we have downloaded the full detailed commentary. 

One improvement to the above method is to use concurrent requests. Since we know the size of the innings, and how many overs are covered in each paginated section, we can concurrently request and download all of these commentary jsons and then order them locally. This saves a magnitude of time when downloading the data. Finally, to further save time on subsequent commentary grabs, we will save the commentary locally in a json format. This prevents us from having to download all the commentary every time we initialize the match_data object.

### Commentary Analysis

We are attempting to classify the result of the ball based off of the comentary for that particular delivery. That is, if there was a four or wicket, etc. on a partiular delivery, can we find that out based on the commentary of that delivery? My initial thoughts to identify the deliveries was to simply search in the commentary if certain words occurred more frequently. This would make sense, as if a batsman hit a boundary, it is logical to assume that the commentary for that ball would include the words boundary, four, 4 etc. 

This seemed not to be the case however. None of the delivery types, four, six or wicket, seemed to have a consistent universal word that appeared to indicate the nature of the delivery. This meant that some sort of machine learning or classification model needed to be employed. Another reason for trying to classify this data is because there are already distinct labels created for the commentary, therefore it is a reletively less time intensive way of learning to create accurate classification models, which can then be used when we are trying to classify other things, such as the type of shot played, or if the player was in control of the shot or not (labels will need to be created in this situation as these labels are not available at the moment).

#### v0.0 Model
This initial verion of the model is a simple SVM classifier with minimal modifications from the standard model provided on sklearn. To build this model, first we create a dictionary of words that are present in commentary, once we have this dictionary, the commentaryTextItems are then encoded into a numerical format. This can be a one-hot encoded format, where each piece of data is tranformed  into a vector the length of the dictionary, where we encode a 1 if the word is presentin the data, or 0 if not. The problem with this is that if we have a large dictionary length, then the vectors that are considered will be very sparse. We can use sparse matrix computation to save resources when dealing with such data. Other than one-hot encoding, we can also create a similar vetor, but then count the frequency of word occurances, so we will have a non-zero number in the vector if the word is present (related to the frequency of occurance) and 0 if the word is not present. In our case we are using this frequency vector method. In addition we are also removing very common words such as 'a', 'at' , 'I', etc. Finally, rather than counting occurances, we count the frequency of the word, and also weight the frequency higher if the word appears less in all the documents (i.e. inverse document frequency. This means that more unique words are weighted higher and therfore more deterministic when we classify).

Unfortinately this model turned out to not work well at all. Due to the high praportion of 'noEvent' commentary present, the model could achieve a high accuracy simply by guessing 'noEvent' every time. This meant that though the model was highly accurate, it never actually made any predictions. For the next iteration of the model, we can either beakdown the 'noEvent' balls into 'dotBalls' , 'one', 'two', etc. OR we can attempt to employ some techniques to combat against imbalanced classification. 