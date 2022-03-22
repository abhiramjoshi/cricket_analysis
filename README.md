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
