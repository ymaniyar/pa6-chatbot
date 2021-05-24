# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import numpy as np
import random
from porter_stemmer import PorterStemmer
from collections import defaultdict


NOT_SURE = "Hmm...not sure about that. Please tell me about a movie you have watched."

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
        self.formatted_names = {'with_year': [], 'without_year':[]}
        self.titles_no_year = set()
        self.title_to_idx = defaultdict(list)
        self.genres_map = defaultdict(list)
        for i, title in enumerate(self.titles):
            name = title[0]
            year_search = re.search(r'( \(\d+(?:-\d*)?\))', name)
            year = None
            if year_search:
                year = year_search.group(1)

            alternates = re.search(r"((?:\([^\)]+\)\s*)+)(?:\(\d\d\d\d\))", name)
            if alternates:
                main_name = name.split(' (')[0]
                names = [main_name]
                alts = alternates.group(1).split(') ')[:-1]
                for alt in alts:
                    alt = alt[1:]
                    aka = re.search(r"(\s*a\.k\.a\.\s*)", alt)
                    if aka:
                        alt = alt.replace(aka.group(1), "")
                    names.append(alt)
            else:
                if not year:
                    names = [name]
                else:
                    names = [name.replace(year, "")]
            new_names = []
            new_names_with_year = []
            for name in names:
                article = re.search(r"(, (?:An|The|A|La|Le|Les|Il|L\'|I))$",name)
                if article:
                    art = article.group(1)
                    name = name.replace(art,"")
                    name = art[2:] + " " + name
                if not year:
                    year = ""
                # new_title = [name.lower()+year,title[1].lower()]
                # new_titles.append(new_title)
                new_names.append(name)
                name_with_year = name+year
                new_names_with_year.append(name_with_year)
                name = name.lower()
                name_with_year = name_with_year.lower()
                self.genres_map[name_with_year].append(title[1])
                self.title_to_idx[name_with_year].append(i)
                self.title_to_idx[name].append(i)
                self.titles_no_year.add(name.lower())
                # self.titles_no_year_set = set(self.titles_no_year)
                # self.titles_no_year.append(re.split(r'( \(\d{4}\))',name.lower())[0])
            self.formatted_names['without_year'].append(new_names)
            self.formatted_names['with_year'].append(new_names_with_year)

        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # integers easier for computation than strings, stems are better than specific words
        self.sentiment = {self.p.stem(key): (1 if val == 'pos' else -1) for key, val in self.sentiment.items()}
        self.negations = ['not',
                            'no',
                            'none',
                            'nobody',
                            'nothing',
                            'neither',
                            'nowhere',
                            'never',
                            "won't",
                            "can't",
                            "didn't",
                            "couldn't",
                            "wouldn't",
                            "shouldn't",
                            "wont",
                            "cant",
                            "didnt",
                            "couldnt",
                            "wouldnt",
                            "shouldnt"]
        self.articles = ['a','an','the']
        self.yes = ['yes',
                    'yup',
                    'yeah',
                    'yea',
                    'yah',
                    'ya',
                    'sure',
                    'mhm',
                    'yurp',
                    'definitely',
                    'absolutely',
                    'i did',
                    'i do']
        self.no = ['no',
                    'nope',
                    'not really',
                    'na',
                    'nah',
                    'i did not',
                    'i do not',
                    'no way',
                    'definitely not',
                    'absolutely not',
                    "i didn't",
                    "i didnt"]
        # chatbot logistics
        self.user_ratings = np.zeros(len(self.titles))
        self.ratings_counter = 0
        self.num_ratings_needed = 5
        self.prev_q_data = None


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
        # Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hi! Tell me about some movies you have watched."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "See you later!"

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
        line = self.preprocess(line)
        if self.creative:
            # extracts type of user input, tense if necessary for chatbot response,
            # and content of input
            root, tense, content = self.parse_line(line)
            movie_names = self.extract_titles(line)
            print(root, tense, content)

            # user wants recs
            if root and root == 'done':
                response = self.deliver_recs()
            # user is responding to a chatbot's question
            elif root and root in ['yes', 'no']:
                if self.prev_q_data:
                    response = self.process_prev_q_response(root)
                else:
                    response = NOT_SURE
            # user is asking a question
            elif root and root in ['can', 'what']:
                if root == 'can':
                    if tense == 'you':
                        response = f'Feel free to {content} if {tense} want!'
                    elif tense == 'I':
                        response = f'{tense} might be able to {content} in the near future.'
                elif root == 'what':
                    response = f'I am not exactly sure what {content} {tense}.'

            # user has mispelled movies
            



            # user has said something about movies
            elif len(movie_names) > 0:
                for movie_name in movie_names:
                    movie_idxs = self.find_movies_by_title(movie_name)
                    # TODO: DISAMBIGUATION HERE to find correct movie_idx
                    # movie_idx = self.find_movies_by_title(movie_name)[0]
                    # sentiment = self.extract_sentiment(line)
                    # response = self.process_movie_rating(movie_idx, sentiment)
            # catch all to respond to the emotion of user's input
            else:
                response = self.identify_emotion(line)
        else:
            # extracts type of user input, tense if necessary for chatbot response,
            # and content of input
            root, tense, content = self.parse_line(line)
            movie_names = self.extract_titles(line)
            print(root, tense, content)
            # user wants recs
            if root == 'done':
                response = self.deliver_recs()
            # user is responding to a question
            elif root in ['yes', 'no']:
                if self.prev_q_data:
                    response = self.process_prev_q_response(root)
                else:
                    response = NOT_SURE
            # user has said something about movies
            elif len(movie_names) > 0:
                movie_name = movie_names[0]
                movie_idx = self.find_movies_by_title(movie_name)[0]
                sentiment = self.extract_sentiment(line)
                response = self.process_movie_rating(movie_idx, sentiment)
            # catch all to respond to the emotion of user's input
            else:
                response = self.identify_emotion(line)



        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response


    def process_movie_rating(self, movie_idx, sentiment):
        """
        Checks whether some sentiment has been expressed towards movie,
        adds rating if so, and if not, asks user whether they liked
        or disliked the movie.

        :param movie_idx: single index of movie
        :param sentiment: rating towards movie at movie_idx
        :returns: response
                either notes that rating has been added,
                clarification question otherwise
        """
        formatted_movie_name = self.formatted_names['without_year'][movie_idx][0]
        if sentiment != 0:
            response = self.add_rating(movie_idx, sentiment)
        else:
            response = f"Did you like {formatted_movie_name}?"
            self.prev_q_data = {'type': 'ask_if_liked', 'movie_idx': movie_idx}
        return response


    def deliver_recs(self):
        """
        Checks whether the user has supplied the necessary minimum of ratings, and
        lists movie recommendations if so. Otherwise, reprompts user for more ratings.

        :returns: response
            either notes that not enough ratings have been given,
            otherwise lists recs

        """
        if self.ratings_counter < self.num_ratings_needed:
            response = f'Sorry, we need at least {self.num_ratings_needed} before we can give you recommendations! Tell me about a movie you have watched.'
        else:
            movie_recs = self.recommend(self.user_ratings, self.ratings)
            movie_recs_str = ""
            for i, idx in enumerate(movie_recs):
                movie_recs_str += f"\n{self.formatted_names['with_year'][idx][0]}"
            response = f'Perfect! Here are some movies I would recommend to you based on your ratings: {movie_recs_str}'
        return response

    def process_prev_q_response(self, response):
        """
        Processes a user's response based on the type of question the chatbot asked.

        :param response: some response to chatbot question (e.g. 'yes', 'no')
        :returns: response

        """
        if self.prev_q_data['type'] == 'ask_if_liked':
            movie_idx = self.prev_q_data['movie_idx']
            if response == 'yes':
                sentiment = 1
            else:
                sentiment = -1
            response = self.add_rating(movie_idx, sentiment)
            self.prev_q_data = None
        else:
            response = NOT_SURE
        return response

    def add_rating(self, movie_idx, sentiment):
        """
        Adds/updates rating matrix. If the movie has not already been rated, increments
        self.ratings counter. Responds with confirmation.

        :param movie_idx: single index of movie
        :param sentiment: rating towards movie at movie_idx
        :returns: response
                explicit confirm that rating has been recorded for given movie.

        """
        assert(sentiment != 0)
        formatted_movie_name = self.formatted_names['without_year'][movie_idx][0]
        if self.user_ratings[movie_idx] == 0:
            self.ratings_counter += 1
        self.user_ratings[movie_idx] = sentiment
        if sentiment > 0:
            sentiment_str = 'liked'
        else:
            sentiment_str = 'did not like'
        response = f'Okay, I have noted that you {sentiment_str} {formatted_movie_name}.'
        if self.ratings_counter >= self.num_ratings_needed:
            response += " Say \'done\' when you would like to hear my movie recommendations for you!"
        return response

    def correct_tense(self, content):
        """
        Reformats 'content' of message to mirror it back to the user.
        e.g. 'i am sad' --> 'you are sad'
        e.g. 'tell me about some movies that you like' --> 'tell you about some movies that i like'

        :param content: content of user input (excludes question words, ending punctuation)
        :returns: content

        """
        content = re.sub(r'( me | me$|^me | i | i$|^i )', ' #you# ', content)
        content = re.sub(r'( am | am$|^am )', ' are ', content)
        content = re.sub(r'( you | you$|^you )', ' I ', content)
        content = re.sub(r'( #you# | #you#$|^#you# )', ' you ', content)
        content = content.strip()
        return content

    def parse_line(self, line):
        """
        Parses line to extract the type of user input (response, question, movie rating, etc.),
        accompanying information about the tense (first person, second person), and content of
        message

        :param content: content of user input (excludes question words, ending punctuation)
        :returns: (root, tense, content)
            root: type of input ('yes', 'no', 'done', 'can_u', etc.)
            tense: used if necessary for grammatically correct response
            content: content of user input (excludes question words, ending punctuation)

        """
        if line in self.yes:
            return ('yes', None, line)
        elif line in self.no:
            return ('no', None, line)
        elif line == 'done':
            return ('done', None, line)
        ending_punc = re.search(r'([\.\,\?\!]+)$', line)
        if ending_punc:
            line = line.replace(ending_punc.group(1), "")
        can_i = re.search(r'^((?:can|could|would)\s+i)', line)
        can_u = re.search(r'^((?:can|could|would)\s+(?:you|u))', line)
        what_is = re.search(r'^((?:what|how)\s+is)', line)
        what_are = re.search(r'^((?:what|how)\s+are)', line)
        if can_i:
            content = line.replace(can_i.group(1), "")
            content = self.correct_tense(content)
            return ('can', 'you', content)
        elif can_u:
            content = line.replace(can_u.group(1), "").strip()
            content = self.correct_tense(content)
            return ('can', 'I', content)
        elif what_is:
            content = line.replace(what_is.group(1), "").strip()
            return ('what', 'is', content)
        elif what_are:
            content = line.replace(what_are.group(1), "").strip()
            return ('what', 'are', content)
        else:
            return (None, None, line)

    def preprocess(self, text):
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
        if self.creative:
            text = text.lower().strip()
        else:
            text = text.strip()
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
        # if year:
        #     indices = [i for i, x in enumerate(self.titles) if x[0] == title]
        # else:
        #     indices = [i for i, x in enumerate(self.titles_no_year) if x == title]
        indices = self.title_to_idx[title]
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
            # print(stem)
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
        sentiment_first = self.extract_sentiment(descriptions[0].lower())
        sentiment_second = self.extract_sentiment(descriptions[1].lower())
        if sentiment_second == 0:
            to_send = descriptions[1].lower() + descriptions[0].lower()
            sentiment_second = self.extract_sentiment(to_send)
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
        for movie in self.titles_no_year:
            dist = self.min_edit_distance(title.lower(),movie.lower(),max_distance)
            if dist != -1:
                if dist < min:
                    titles.clear()
                    titles.extend(self.title_to_idx[movie])
                    min = dist
                elif dist == min:
                    titles.extend(self.title_to_idx[movie])
        return np.unique(titles)

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


    def identify_emotion(self, line):
        # ADONIS
        # 'line' should only be passed to this function if movie info could not be extracted from it
        # 'line' should be all lowercase from pre-processing
        # synonyms are top results from thesaurus.com
        # uses explicit confirmation
        # does not respond to negations
        emotions = {
            'angry': {'angry','bitter','enraged','exasperated','furious','heated','impassioned',
                              'indignant','irate','irritable','irritated','offended','outraged',
                              'resentful','sullen','uptight'},
            'sad': {'sad','dismal','heartbroken','melancholic','depressed','mournful',
                    'pessimistic','somber','sorrowful','sorry','wistful'},
            'happy': {'happy','cheerful','contented','delighted','ecstatic','elated','glad',
                      'joyful','joyous','jubilant','lively','merry','overjoyed','peaceful','pleasant',
                      'pleased','thrilled','euphoric'}
        }

        response = "Sorry, I didn't catch that. Could you rephrase?"
        found = re.search(r'(?:^i am | i am |^i\'m | i\'m |^im | im | im$)(?: |\w)+', line)
        if found:
            sentiment = self.extract_sentiment(found.group())
            for root in emotions: # assumes only one emotion is present in response
                for synonym in emotions[root]:
                    if synonym in found.group():
                        print(sentiment)
                        if root in ('angry','sad') and sentiment != 1: # make sure sentiment doesn't conflict root
                            options = [f"Oh! Did I make you {root}? I aplogize.",
                                       f"Yikes! I may have caused you to become {root}. Please forgive me.",
                                       f"You're {root}? I'm sorry to have made you feel that way.",
                                       f"I didn't mean to make you {root}. I hope you can forgive me."]
                            response = random.choice(options)
                        elif root == 'happy' and sentiment != -1:
                            options = [f"Great! It's good to hear that you're {root}.",
                                       f"Glad you're feeling {root}!",
                                       f"Amazing! I'm {root} that you're {root} :)",
                                       f"Yay! I hope my movie recommendations make you {root} too.",
                                       f"Nice! I hope that you contine you to be {root}."]
                            response = random.choice(options)
        return response



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
        clarification = clarification.lower()
        single_digit = False
        
        if clarification.isdigit() and len(clarification) == 1: # minimize false positives -> search for installment
            clarification = ' ' + clarification + ' '
            single_digit = True
            
        if clarification.startswith(('19', '20')) and len(clarification) == 4: # minimize false positives -> search for year
            clarification = '(' + clarification + ')'
            
        narrower = [i for i in candidates if clarification in self.titles[i][0]]
        
        if single_digit and len(narrower) == 0: # assumes digit is 1-index in list if installment search not successful
            i = int(clarification.strip())
            if 0 < i <= len(candidates):
                narrower = [candidates[i - 1]]
                
        if 'most recent' in clarification or 'newest' in clarification: # assume candidates in list can be distinguished by year
            narrower = [max(candidates, key=lambda i: int(re.search(r'\(\d{4}\)', self.titles[i][0]).group()[1:-1]))]
            
        if 'least recent' in clarification or 'oldest' in clarification: # assume candidates in list can be distinguished by year
            narrower = [min(candidates, key=lambda i: int(re.search(r'\(\d{4}\)', self.titles[i][0]).group()[1:-1]))]
        
             # try ordinal index representation
        if 'first' in clarification:
            narrower = [candidates[0]]
        if 'second' in clarification:
            narrower = [candidates[1]]
        if 'third' in clarification:
            narrower = [candidates[2]]
        if 'fourth' in clarification:
            narrower = [candidates[3]]
        if 'fifth' in clarification:
            narrower = [candidates[4]]
                
        return narrower

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
