# Customer-Sentiment-Analysis
Presentation link: http://youtu.be/3I0-YX48Avs?hd=1

**Main idea:**

The major idea is to analyse the customer feedback (whether it is positive or negative and predicting the probability of statement as well). For entering a statement a User Interface is created using Html and css, output is given according to a trained model which is trained using the concepts of neural networks then using the flask.

**Approach:**

Initially to train the model we need some data, I took the data from kaggle (imdb data set which has about 50,000 entries). Then there were several inconsistencies in the data and also the stop word, Hence I removed the stop word (like is, and) using NTLK corpus and the other inconsistencies were removed using replace function in python. Now we have read the data and created the dataframe using pandas

As we know we need to train and test the model, we import train_test_split from sklearn and now we write a function for data cleaning of the input data and also convert the word “positive “ and “negative” to 1 and 0 respectively.

Now we assign x_train, y_train, x_test and y_test and the encoding is done in which word tokenizing is done and we also use the inbuilt function one_hot (assignment of numbers to words) 

We use the sequential model from training and testing of data, so in the sequential model the most important thing is padding, we pad the data which is stored in the form of lists and now comes the most important thing which is making the sequential model.
For sequential model we use 5 layers which are LSTM (long short memory) for sequence prediction and is most important layer, then we use embedding layer for input, dense layer which is used to take data from previous layers neurons and we use 2 dropout layer to drop the number of neutrons and make model 

Now test by taking a random input statement and test it whether it is positive or not, model.predict.class gives the output and model.predict gives us the probability of statement being positive or negative respectively .

Finally, the .h5 file will be created which will be used in application.py file

**User Interface and Flask:**

In application of py we use various features of flask like put and get for data getting and outputting the result accordingly and we connect the trained.h5 file and home.html file here, so we can call this is the most import part of the entire project.

Next is the user interface in which we enter the text ( a movie review) and we get whether it is positive or negative and probability of it being positive of negative along with an emoji (which was added in a folder in app.py file).
The code for HTML file is attached in codelink.

**Conclusion:**

The sequential model gives a good output and also the data set (which is used for training the model is really useful as it has 50,000 entries). I am looking forward to improving the accuracy of that model (currently between 88% to 95%) and also improving the user interface.


