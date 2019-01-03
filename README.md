MACHINE LEARNING
	
	* FEATURES
	  -Features are those properties of a problem based on which you would like to predict results.
	  -For example,
		 ~In image processing, features can be gradient magnitude, color, grayscale intensity, edges, areas, and more. 
	 	 ~In speech recognition, features can be sound lengths, noise level, noise ratios, and more.
	 	 ~In spam-fighting initiatives, features are abundant. They can be IP location, text structure, frequency of certain words, or certain email headers

	* TYPES
	  1.Supervised Learning: We have instances of any target or outcome variable to predict. 
		~COMMON ALGOS:  -Nearest Neighbor
			        -Naive Bayes
			        -Decision Trees
				-Linear Regression
				-Support Vector Machines (SVM)
	  2.Unsupervised Learning: We do not have any target or outcome variable to predict. It is also called as clustering.
				-K-means
				-Hierarchical Clustering(Divisive, Agglomerative)
	  3.Semi-supervised Learning: Combination of supervised and unsupervised learning, typically containing a small amount of labeled data with a large amount of unlabeled data.

	 -> Classification:- Classification is the process of predicting the class of given data points. So the computer program learns from the data input given to it and then uses this learning to classify new observation.
			  Common algos used:- Logistic Regression, Naïve Bayes, K-Nearest Neighbours, Decision Tree,  Random Forest,  Support Vector Machine. 
	 -> Regression:- It is the process of predicting values of a desired target quantity when the target quantity is continuous.
			   Common algos used:- Simple linear regression, Support vector machines, multivariate regression algorithm.

	*TOOLS
	-scikit-learn
	-weka
	-Jupyter Notebook
	-numpy, pandas, matplotlib

DEEP LEARNING
	
	~ Deep learning algorithms heavily depend on high-end machines, so the basic requirements of deep learning algorithm include GPUs.
	~ Deep learning algorithms try to learn high-level features from data.
	~ When solving a problem using traditional machine learning algorithm, it is generally recommended to break the problem down into different parts, solve them individually and combine them to get the result. 
		Deep learning in contrast advocates to solve the problem end-to-end.
	~ Deep learning algorithms takes a long time to train.
	~ It is particularly difficult to interpret the reasoning behind results of deep learning algorithms.

	* TYPES
	-Deep Boltzmann Machine (DBM)
	-Deep Belief Networks (DBN)
	-Convolutional Neural Network (CNN)
	-Stacked Auto-Encoders


NATURAL LANGUAGE PROCESSING(NLP)

*NLP is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. 

	~TOPICS OF NLP:-

		-Content categorization: A linguistic-based document summary, including search and indexing, content alerts and duplication detection.
	
		-Topic discovery and modeling: Accurately capture the meaning and themes in text collections, and apply advanced analytics to text, like optimization and forecasting.

		-Contextual extraction: Automatically pull structured information from text-based sources.

		-Sentiment analysis: Identifying the mood or subjective opinions within large amounts of text, including average sentiment and opinion mining. 

		-Speech-to-text and text-to-speech conversion: Transforming voice commands into written text, and vice versa. 
	
		-Document summarization: Automatically generating synopses of large bodies of text.

		-Machine translation: Automatic translation of text or speech from one language to another.

		-Identifying scammers: Natural Language Processing uses spam filters. This is a commonly used defense mechanism much needed to identify spam messages and emails.

		-Question Answering: A QA application is a system capable of logically answering a valid human request. NLP has the capability to understand the human languages either in text-only format or spoken dialogue.


	~MAIN APPROACHES IN NLP:-
		
		1.Rule based methods:
			
			*Regular expressions
			*Context Free Grammers

		2.Probablistic modeling and Machine learning:
		
			*Likelihood Maximization
			*Linear Classifiers
		
		3.Deep Learning
			
			*Recurrent Neural Networks
			*Convolutional Neural Networks
		

	~TEXT PREPROCESSING:-

		1.Tokenization: It is a process that splits an input sequence into so-called tokens.
						-nltk.tokenize

		2.Token Normalization:

				*Stemming: Stemming is a process of removing and replacing suffixes to get to the root form of the word, which is called the stem.		
						-Porter's Stemmer: nltk.stem.PorterStemmer

				*Lemmatization: Returns the base or dictionary form of a word, known as lemma.
						-Wordnet Lemmatizer: nltk.stem.WordNetLemmatizer


	~WORD EMBEDDING:-
		-Word Embeddings are the texts converted into numbers and there may be different numerical representations of the same text

			*Bag Of Words(BOW): It is a sum of sparse one-hot-encoders. Uses text vectorization to count occurrences of a particular token in our text.
					
			*N-Grams of BOW:  An n-gram is a contiguous sequence of n items from a given sample of text or speech.

			*TF-IDF Vectorizers as BOW: tfidf(t,d,D) = tf(t,d)*idf(t,D) where, tf(t,d) is frequency of term(t) in document(d) and idf is number of documents where term t appears.


	~TEXT CLASSIFICATION MODELS ON WORD EMBEDDING:-
		
		*Linear Models: Can be used over bag of 1,2-grams with TF-IDF 

				-Logistic Regression
				-Naive Bayes
				-SVM

		*Hashing Techniques: Used for mapping n-grams to feature indices. 

				Example: { n-gram -> hash(n-gram) % 2^20 } 

				-So can be used for huge datasets as it maps data into a pre-specified size.
						-sklearn.feature_extraction.text.HashingVectorizer

		*Word2Vec neural network techniques: It is a two layer neural network used to generate word embeddings given a text corpus. 
						     The word2vec objective function causes the words that occur in similar contexts to have similar embeddings.

				-> CBOW(Continous Bag Of Words) Model: This method takes the context of each word as the input and tries to predict the word corresponding to the context.
								       
				-> Skip-Gram model: Predict the context word from the given target.
		
					Example: "Hope can set you free"
						In CBOW model, for window size 3 it uses "hope" and "set" to predict "can".
						In Skip-Gram model, for window size 3 it uses "can" to predict "hope" and "set".
	