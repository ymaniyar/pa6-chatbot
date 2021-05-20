# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import numpy as np
from porter_stemmer import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'
        self.creative = creative
        self.p = PorterStemmer()

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        # convert movie titles to lowercase
        new_titles = []
        self.titles_no_year = []
        for title in self.titles:
            name = title[0]
            article = re.search(r"(, (?:An|The|A))",name)
            if article:
                art = article.group(1)
                name = name.replace(art,"")
                name = art[2:] + " " + name
            new_title = [name.lower(),title[1].lower()]
            new_titles.append(new_title)
            self.titles_no_year.append(re.split(r'( \(\d{4}\))',name.lower())[0])
        self.titles = new_titles

        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        # integers easier for computation than strings, stems are better than specific words
        self.sentiment = {self.p.stem(key): (1 if val == 'pos' else -1) for key, val in self.sentiment.items()}
        self.negations = ['not', 'no', 'none', 'nobody', 'nothing', 'neither', 'nowhere', "won't", 'never', "can't", "didn't", "couldn't", "wouldn't", "shouldn't"]
        self.articles = ['a','an','the']


        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        # 1. remove non-alphanumeric characters
        # 2. convert to lowercase

        #text = re.sub(r'[^\w\s]', '', text.lower());

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        # SOFIA
        """
        Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]s

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """


        ##Question mark makes expression lazy rather than greedy (stops as soon as it finds something).
        matches=re.findall(r'\"(.+?)\"',preprocessed_input)
        if len(matches) == 0:
            matches = []
            words = preprocessed_input.lower().split()
            for i in range(len(words)):
                base = words[i]
                for j in range(i+1,len(words)+1):
                    if base in self.titles_no_year:
                        matches.append(base)
                    elif base[:len(base)-1] in self.titles_no_year:
                        matches.append(base[:len(base)-1])
                    if j < len(words):
                        base = base + " " + words[j]
        return matches

    def find_movies_by_title(self, title):
        # SOFIA
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        title = title.lower()
        year = re.search('(\(\d{4}\))',title)
        if year:
            indices = [i for i, x in enumerate(self.titles) if x[0] == title]
        else:
            indices = [i for i, x in enumerate(self.titles_no_year) if x == title]
        return indices

    def extract_sentiment(self, preprocessed_input, simple=True):
        # TODO: ADONIS
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        num_words = len(preprocessed_input.split())
        stemmed_input = [self.p.stem(word) for word in preprocessed_input.split()]

        scores = []
        multiplier = 1
        negation = 1
        for stem in stemmed_input:
            base = self.sentiment[stem] if stem in self.sentiment else 0
            scores.append(base*multiplier*negation)
            multiplier = 1
            if stem in [self.p.stem('very'), self.p.stem('really')]:
                multiplier = 2 # increase score of following word
            if stem in self.negations:
                negation *= -1
        if sum(scores) > 0:
            return 1
        elif sum(scores) == 0:
            return 0
        else:
            return -1

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        titles = self.extract_titles(preprocessed_input)
        descriptions = preprocessed_input.split(titles[0])
        sentiment_first = self.extract_sentiment(re.sub(r'[^\w\s]', '', descriptions[0].lower()))
        sentiment_second = self.extract_sentiment(re.sub(r'[^\w\s]', '', descriptions[1].lower()))
        if sentiment_second == 0:
            sentiment_second = sentiment_first
        return([(titles[0],sentiment_first),(titles[1],sentiment_second)])

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        min = max_distance
        titles = []
        for i in range(len(self.titles_no_year)):
            movie = self.titles_no_year[i]
            dist = self.min_edit_distance(title.lower(),movie.lower(),max_distance)
            if dist != -1:
                if dist < min:
                    titles.clear()
                    titles.append(i)
                    min = dist
                elif dist == min:
                    titles.append(i)
        return titles

    def min_edit_distance(self,source,target,max):
        n = len(source)
        m = len(target)
        d = np.zeros((n+1,m+1))
        for i in range(1,n+1):
            d[i,0] = d[i-1,0] + 1
        for i in range(1,m+1):
            d[0,i] = d[0,i-1] + 1
        for i in range(1,n+1):
            for j in range(1,m+1):
                ins_cost = 0 if source[i-1]==target[j-1] else 2
                d[i,j] = min(d[i-1,j]+1,d[i-1,j-1]+ins_cost,d[i,j-1]+1)
                if min(d[:,j]) > max:
                    return -1
                    break
        return d[n,m]


    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        pass

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.

        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[ratings == 0] = 0

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a numpy array or
        :param v: another vector, as a numpy array

        :returns: the cosine similarity between the two vectors

        if 1D vectors given, returns scalar cosine similarity between arrays
        if matrices P and Q given, returns cosine similarity matrix S
            where S[i, j] is cosine similarity between ith row of P and jth row of Q
        """

        norm_u = np.linalg.norm(u, axis=-1, keepdims=True)
        norm_v = np.linalg.norm(v, axis=0, keepdims=True)
        denoms = np.dot(norm_u, norm_v)

        # avoid dividing by 0
        if np.isscalar(denoms):
            if denoms == 0:
                denoms = 1
        else:
            denoms[denoms == 0] = 1

        similarity = np.matmul(u, v, dtype=float)/denoms
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        # YASH
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # Implement a recommendation function that takes a vector              #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []

        # similarity matrix of shape (num_movies, num_movies) where sims[i, j]
        # is cosine similarity between movies i and j
        sims = self.similarity(ratings_matrix, ratings_matrix.T)

        # scores[i] = score for movie i, which is sum of user ratings weighted by similarity to movie i;
        # note that user ratings of 0 effectively do not contribute to score
        scores = np.sum(np.multiply(sims, user_ratings), axis=-1)

        # we do not want to include movies that the user has already seen in final recommendations
        scores[user_ratings != 0] = np.NINF

        # recs are our k highest scoring movies
        recommendations = list(np.argsort(-1*scores)[:k])

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
