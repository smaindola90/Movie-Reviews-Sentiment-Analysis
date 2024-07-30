import re
import nltk
import string
from flask import Flask, render_template, request
import pickle
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    '''This function is used for preprocessing the text.'''

    # Lowercasing the text
    new_text = text.lower()
    
    # Removing HTML tags
    pattern = re.compile('<.*?>')
    new_text = pattern.sub(r'', new_text)
    
    # Removing URLs from the text
    pattern = re.compile(r'https?://\S+|www\.\S+')
    new_text = pattern.sub(r'', new_text)

    # Removing emails from the text
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    new_text = pattern.sub(r'', new_text)
    
    # Removing formatting from the text
    new_text = new_text.replace('! :\x8d','').replace('\x08','').replace('\x9e','').replace('\x8e','').replace('\x97',' ')\
                                 .replace('\t',' ').replace('\xa0',' ').replace('\x10','').replace('\x80','').replace('\x96','').replace('\x84',' ')\
                                 .replace('\x85',' ').replace('\x91',' ').replace('\x95','').replace('\uf0b7','').replace('\xad','').replace('\x9a','')
    
    # Removing special characters from the text
    new_text = new_text.replace('★','').replace('»',' ').replace('«',' ').replace('▼',' ').replace('…',' ').replace('§','')\
                                 .replace('¡','').replace('¦',"'").replace('®','').replace('¨',' ').replace('¿','').replace('，',' ').replace('、',' ')\
                                 .replace('·',' ').replace('″','').replace('“','').replace('–','').replace('”','').replace('‘',"'").replace('´',"'")\
                                 .replace('’',"'")
    
    # Tokenization
    tokens = nltk.word_tokenize(new_text)
    
    # Removing punctuations
    punctuations = string.punctuation
    new_tokens = []
    for token in tokens:
        if any(char not in punctuations for char in token):
            new_tokens.append(token)
    tokens = new_tokens[:]

    # Removing stop words
    stop_words = nltk.corpus.stopwords.words('english')
    new_tokens = []
    for token in tokens:
        if token not in stop_words:
            new_tokens.append(token)
    tokens = new_tokens[:]

    # Removing additional unnecessary tokens
    unnecessary_tokens = ["'s", "n't", "'ve", "'m", "'re", "'ll", "'d", "'n", "", "’", "–", "“", ".the", "'the", "i.e", "i.e.", "e.g", ".i", "", 
                          "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i.", "j.", "k.", 
                          "l.", "m.", "n.", "o.", "p.", "q.", "r.", "s.", "t.", "u.", "v.", "w.", "x.", "y.", "z.", "mr.", "dr.", "ms.", "jr.", "mrs.", 
                          "1.a.", "it.i", ".it", ".if", "l.a.", ".this", ".in", "-the", "it´s", "lt.", ".as", "co.", "'you", "w/", "-and", ".and", 
                          "it.the", "'em", "st.", ".there", "'it"]
    tokens = [token for token in tokens if token not in unnecessary_tokens]

    # Stemming
    ps = nltk.stem.porter.PorterStemmer()
    new_tokens = []
    for token in tokens:
        new_tokens.append(ps.stem(token))
    return ' '.join(new_tokens)



app = Flask(__name__)
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

# Get user input and then predict and return the output to user
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        input_text = [x for x in request.form.values()][0]
        # Preprocessing the text
        processed_text = [preprocess_text(input_text)]
        # Vectorization
        x = vectorizer.transform(processed_text).toarray()
        # Prediction
        prediction = model.predict(x)
        prediction_proba = model.predict_proba(x)
        prediction_text = "positive" if prediction==1 else "negative"

        # Return the html page and show the output
        return render_template('index.html', prediction_text=f"The sentiment of the review is {prediction_text} with {round(max(prediction_proba[0])*100)}% confidence.")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
