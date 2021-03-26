Here you can find the list of different types of tokens used in preprocessing textual information and file names.
I have used these dictionaries when cleaning different parts of the data.
For instance, using `contractions.txt` list, one can convert **ain't** to **are not**. 
Or more specifically for Sofware Engineering concepts and tokens,
one can convert **async** to **asynchronous**, and **bg** to **background** using **SE_abbr.txt** list.
I also change SE topics such as **c++** to **cplusplus**, **c#** to **csharp**, etc. before removing punctuations from the text (see `SE_topics.txt` list).
