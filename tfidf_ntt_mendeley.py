import streamlit as st
from Bio import Entrez
import pandas as pd
import bibtexparser
import re
from typing import List
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from bibtexparser.customization import author, convert_to_unicode, doi
from bibtexparser.bparser import BibTexParser
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

class SpacyPreprocessor:
    def __init__(
        self,
        spacy_model=None,
        remove_numbers=False,
        remove_special=True,
        pos_to_remove=None,
        remove_stopwords=False,
        lemmatize=False,
    ):
        """
        Preprocesses text using spaCy
        :param remove_numbers: Whether to remove numbers from text
        :param remove_stopwords: Whether to remove stopwords from text
        :param remove_special: Whether to remove special characters (including numbers)
        :param pos_to_remove: list of PoS tags to remove
        :param lemmatize:  Whether to apply lemmatization
        """

        self._remove_numbers = remove_numbers
        self._pos_to_remove = pos_to_remove
        self._remove_stopwords = remove_stopwords
        self._remove_special = remove_special
        self._lemmatize = lemmatize

        if not spacy_model:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model


    
    # @staticmethod
    # def load_model(model="en_core_web_sm"):
    #     return spacy.load(model, disable=["ner", "parser"])

    def tokenize(self, text) -> List[str]:
        """
        Tokenize text using a spaCy pipeline
        :param text: Text to tokenize
        :return: list of str
        """
        doc = self.model(text)
        return [token.text for token in doc]

    def preprocess_text(self, text) -> str:
        """
        Runs a spaCy pipeline and removes unwanted parts from text
        :param text: text string to clean
        :return: str, clean text
        """
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text_list(self, texts=List[str]) -> List[str]:
        """
        Runs a spaCy pipeline and removes unwantes parts from a list of text.
        Leverages spaCy's `pipe` for faster batch processing.
        :param texts: List of texts to clean
        :return: List of clean texts
        """
        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc: Doc) -> str:

        tokens = []
        # POS Tags removal
        if self._pos_to_remove:
            for token in doc:
                if token.pos_ not in self._pos_to_remove:
                    tokens.append(token)
        else:
            tokens = doc

        # Remove Numbers
        if self._remove_numbers:
            tokens = [
                token for token in tokens if not (token.like_num or token.is_currency)
            ]

        # Remove Stopwords
        if self._remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        # remove unwanted tokens
        tokens = [
            token
            for token in tokens
            if not (
                token.is_punct or token.is_space or token.is_quote or token.is_bracket
            )
        ]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != ""]

        # Lemmatize
        if self._lemmatize:
            text = " ".join([token.lemma_ for token in tokens])
        else:
            text = " ".join([token.text for token in tokens])

        if self._remove_special:
            # Remove non alphabetic characters
            text = re.sub(r"[^a-zA-Z\']", " ", text)
        # remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        text = text.lower()

        return text

def procesPipe(text):
    spacy_model = load_spacy_model()
    preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, 
                                     remove_stopwords=True)
    x = preprocessor.preprocess_text(text)
    return x

def searchpb(search_term, retmax=200, retmode='xml', sort='relevance', mindate = 2018, maxdate = 2021):
    """
    (str) -> Entrez handle
    It receive a query to be searched in pubmed and return the handler of the search
    Optional argumetns are:
    retmax (int) -> maximum number of results to be returned (5000 default)
    retmode (str) -> Format of hits (xml es default)
    sort (str) -> Field to sort by (relevance Default)
    """
    Entrez.email = 'elmaturana@gmail.com'
    Entrez.api_key = '3cfd60b4f78696d27f1c4df78d3fe6f90a09'
    handle = Entrez.esearch(db='pubmed',
                            sort=sort,
                            retmax=retmax,
                            retmode=retmode,
                            term=search_term,
                            mindate = mindate,
                            maxdate = maxdate)
    return Entrez.read(handle)

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'elmaturana@gmail.com'
    Entrez.api_key = '3cfd60b4f78696d27f1c4df78d3fe6f90a09'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

def parse_paperinfo(paperinfo_xml):
    """
    :param paperinfo_xml:
    :return:
    """
    PubmedData = parsePubmedData(paperinfo_xml['PubmedData'])
    article_xml = parseArticle(paperinfo_xml['MedlineCitation']['Article'])
    mayorKeys = parseKeys(paperinfo_xml['MedlineCitation'])
    article_xml['Keys'] = '; '.join(list(set(mayorKeys)))
    try:
         for i, author_xml in enumerate(article_xml['autorlist']):
            if author_xml.attributes['ValidYN'] == 'N':
                print('no valido')
                continue
            autor_dict = parse_author_xml(author_xml)
            article_xml['autorlist'][i] = autor_dict
    except:
        print('ERROR: parsing author {}'.format(author_xml))

    PubmedData.update(article_xml)
    return PubmedData

def parse_author_xml(autor_xml):
    """
    (dict)->dict
    Receive un diccionario con las informaciones de autor proveniente de pubmed xml article

    :param autor_xml:
    :return:
    """
    # Return false if no author information found
    if 'CollectiveName' in autor_xml:
        return
    # try to parse information from XML
    try:
        #get Identifier (only orcid is used now so if they have identifier it should be the first value
        if len(autor_xml['Identifier']) > 0:
            autorID = autor_xml['Identifier'][0]
        else:
            autorID = ''
        #Get the affilaition details from that author, if he had
        if len(autor_xml['AffiliationInfo']) > 0:
            AFFs = ';'.join([affiliationinfo['Affiliation'] for affiliationinfo in autor_xml['AffiliationInfo']])
        else:
            AFFs = ''
        #Retrieving the name information, it is a must and should exist
        autorFN = autor_xml['ForeName']
        autorLN = autor_xml['LastName']
        autorIN = autor_xml['Initials']
        name = autorFN + ' ' + autorLN

        #Start parsing or retrieving information for country, email, company, institute from affiliation
        # country_name, state = find_country(AFFs)
        # emails = parse_email(AFFs)
        data = {'Fname': autorFN, 'Lname': autorLN, 'affiliations': AFFs, 
        'identifier': autorID, 'name': name, 'initials': autorIN}
        return data

    except ValueError:
        print('not possible to get info value error')
        return
    except OSError as err:
        print("OS Error: {0}".format(err))
        return
    except:
        print('error en parsing')
        return

def parsePubmedData(pubmeddata):
    """
    Receive the xml section of PubmedData and return list of ids
    :param pubmeddata:
    :return:
    """
    ids = {x.attributes['IdType']: str(x) for x in pubmeddata['ArticleIdList']}
    return ids

def parseArticle(article_info):
    """

    :param article_info: dictionary from key Article of an Medline citation
    :return (dict): tuple of dictionary with information from paper and autors
    """
    title = article_info['ArticleTitle']
    journal = article_info['Journal']['Title']
    published_date = article_info['Journal']['JournalIssue']['PubDate']
    if 'Year' in published_date:
        published = published_date['Year']
    elif 'MedlineDate' in published_date:
        try:
            published = re.findall(r'\d\d\d\d',published_date['MedlineDate'])[0]
        except:
            published = published_date['MedlineDate'][:4]
    try:
        abstract = '. '.join(article_info['Abstract']['AbstractText'])
    except:
        abstract = ''
    try:
        autorlist = article_info['AuthorList']
    except:
        print('no autors found, jumping next')
        autorlist = []
    return {'abstract': abstract, 'autorlist': autorlist, 'title': title, 'journal': journal,
            'published':published}

def parseMayorKeys(citationInfo):
    keywordList = citationInfo['KeywordList']
    
    if len(keywordList) == 0:
        (mayorMesh, minorMesh) = parseMeshKeys(citationInfo)
        mayorMesh.extend(minorMesh)
        keys = mayorMesh
    else:
        keys = [str(x) for x in keywordList[0] if x.attributes['MajorTopicYN'] == 'Y']
        # keys.extend(mayorMesh)
        # keys.extend(minorMesh)
    return keys

def parseMeshKeys(citationInfo):
    if 'MeshHeadingList' not in citationInfo.keys():
        return [], []
    meshKeys = citationInfo['MeshHeadingList']
    mayorkeys = [str(x['DescriptorName']) for x in meshKeys if x['DescriptorName'].attributes['MajorTopicYN']=='Y']
    minorKeys = [str(x['DescriptorName']) for x in meshKeys if x['DescriptorName'].attributes['MajorTopicYN']=='N']
    for x in [mayorkeys, minorKeys]:
        if x is None:
            x = []
    return mayorkeys, minorKeys

def parseKeys(citationInfo):
    return parseMayorKeys(citationInfo)#, parseMeshKeys(citationInfo)

@st.cache(max_entries=10)
def getParsedArticlesPeriod(name, maxdate, mindate, term):
    if term == 'Author':
        term = '[Author]'
    elif term == 'Title':
        term = '[Title]'
    elif term ==  'Affiliation':
        term = '[Affiliation]'
    elif term ==  'First Author':
        term = '[Author - First]'
    elif term ==  'Last Author':
        term = '[Author - Last]'
    elif term ==  'Keywords':
        term = ''
    query = name.strip(' ') + term
    results = searchpb(query, 100, maxdate = maxdate, mindate = mindate)
    id_list = results['IdList']
    if len(id_list) == 0:
        return 
    papers = fetch_details(id_list)
    articles=[]
    for i, paperinfo in enumerate(papers['PubmedArticle']):
        article = parse_paperinfo(paperinfo)
        articles.append(article)
        if len(articles) == 0:
            return 
    df = pd.DataFrame(articles)
    return df

@st.cache()
def train_model(db):
    db1 = db.copy()
    tfidf = TfidfVectorizer(ngram_range=(1,1))
    X = tfidf.fit_transform(db1['preprocess'])
    return tfidf, X

def customizations(record):
    """Use some functions delivered by the library

    :param record: a record
    :returns: -- customized record
    """
    record = author(record)
    record = convert_to_unicode(record)
    record = doi(record)
    return record

@st.cache()
def load_library():
    db_list = []
    for file in ['Dianthus.bib', 'NanoDSF & Monolith.bib', 'Tycho.bib', 'NanoDSF.bib', 
                 'NanoDSF & Tycho & Monolith.bib', 'Tycho & Monolith.bib', 'Monolith.bib']:
        with open(f"./Data/{file}") as bib_file:
            print(file)
            parser = BibTexParser()
            parser.customization = customizations
            bib_database = bibtexparser.load(bib_file, parser=parser)
            df = pd.DataFrame(bib_database.entries)
            df['instrument'] = file.replace('.bib','')
            db_list.append(df)


    db = pd.concat(db_list).reset_index(drop=True)
    db = db[db['abstract'].notna()].reset_index(drop=True)
    db['author'] = db.author.apply(lambda x: f"{x[0]} et al.")
    db = db[['title', 'abstract', 'instrument', 'keywords', 'mendeley-tags',
     'journal', 'doi', 'author', 'url', 'year']]
    db['preprocess'] = db.apply(lambda row: procesPipe(' '.join(row[['title', 'abstract']]), ), axis=1)
    return db

def searchMendeley(df, tfidf, X):
    """
    giving a df with papers from pubmed, it will join title and abstract, prepocess it and 
    calculate the tf/IFD, for each paper. It will after measure the cosimne similaritie between each of the papers 
    with all the papers from the database. Cosine similarities are ordered by increasing 
    score and returned as a list in a new column

    df1 DataFrrame
    db 
    return df DataFrame
    """
    df1 = df.copy()
    progress_bar = st.progress(0)
    scores = []
    links = []
    df1['process'] = df1.apply(lambda row: procesPipe(' '.join(row[['title','abstract']])), axis=1)
    query_tfidf = tfidf.transform(df1['process'])
    for i, query in enumerate(query_tfidf):
        progress_bar.progress(i/(df1.shape[0]))
        cosine_similarities = linear_kernel(query, X).flatten()
        link= cosine_similarities.argsort()
        links.append(link)
        scores.append(cosine_similarities[link])
    df1['links'] = links
    df1['scores'] = scores
    progress_bar.empty()
    return df1

def df2results(df, db):
    """
    """
    total = []
    for x in range(0,df.shape[0]):
        # st.table(df.iloc[x])
        result = pd.DataFrame({     'score':list(df.iloc[x]['scores']), 
                                    'dbID':list(df.iloc[x]['links']), 
                                    })
        result['queryID']=x 
        result['Title']=df.iloc[x]['title']
        result['Authors']= '; '.join([x['name'] for x in df.iloc[x]['autorlist']])
        result['Abstract']=df.iloc[x]['abstract']
        result['Published']=df.iloc[x]['published']
        result['doi']=df.iloc[x]['doi']
        result['Journal']=df.iloc[x]['journal']
        result['Keys']=df.iloc[x]['Keys']
        result_merged = pd.merge(result, db, left_on='dbID', right_index=True)
        columns = [ 'score', 'dbID', 'queryID', 'Title', 'Abstract', 'Authors', 'Journal', 'Published', 'Keys', 'doi_x',
        'title', 'instrument', 'abstract', 'mendeley-tags', 'journal', 'year', 'doi_y',  'author']
        result_merged = result_merged[columns]
        result_merged.columns = [ 'Score', 'dbID', 'queryID', 'Title', 'Abstract', 'Authors', 'Journal', 'Published', 'Keys', 'doi',
        'DB Paper', 'Instrument', 'DB abstratc', 'Keywords DB', 'DB Journal', 'DB Year', 'DB doi',  'DB Author']
        total.append(result_merged.sort_values('Score', ascending=False))
    return pd.concat(total)
    

@st.cache(max_entries = 10, suppress_st_warning=True)
def pipeline(df, db, tfidf, X):
    df1 = searchMendeley(df, tfidf, X)
    return df2results(df1, db)

def filter_scores(df):
    if technology == 'MST':
        df = df[df['Instrument'].str.contains('Monolith')]
    elif technology == 'nanoDSF':
        df = df[df['Instrument'].str.contains('NanoDSF')]
    elif technology == 'Tycho':
        df = df[df['Instrument'].str.contains('Tycho')]
    return df[df.Score >= cut_off]

def grouped_sugestions(df):
    return df.groupby(['queryID']) #[x for x in grouped_df]


def download_spacy_model(model="en_core_web_sm"):
    print(f"Downloading spaCy model {model}")
    spacy.cli.download(model)
    print(f"Finished downloading model")

@st.cache(allow_output_mutation=True)
def load_spacy_model(name="en_core_web_sm"):
    download_spacy_model()
    return spacy.load(name, disable=["ner", "parser"])

db = load_library()
tfidf, X = train_model(db)

year_from = st.sidebar.number_input('From', value=2019)
year_to = st.sidebar.number_input('to', value=2021)
term = st.sidebar.selectbox('Search By:', ['Author', 'Title', 'Affiliation', 'First Author',
                                            'Last Author', 'Keywords'])
topN = st.sidebar.number_input('Top # results for each paper', value=3)
technology = st.sidebar.selectbox('Filter results by technologies', ['All', 'MST', 'nanoDSF', 'Tycho'])
cut_off = st.sidebar.number_input('Cut Off for similarity score', value=0.20)

st.title('Getting sugestions from Mendeley Library')
st.write('This tool use NLP technique to calculate the similarity between 2 papers title and abstract, '
    'First the model is trained with the mendeley library from NTT, calculating the TF/IDF of the words '
    'that are present in the articles. After that it do the same, using the learned vocabulary for each of the '
    ' articles found on PubMed and calculate the cosine similarity of this 2 vectors.'
    'It will search and author, affiliation or keyword from pubmed, will filter between the years selected ',
    'and for each of the papers will measure the similarities to each the articles in our mendeley database '
    ' and show the results by instrument related',
    'Suggestions, use the first and last name of the author'
    )


name = st.text_input(f"Search for a specific {term}")
if not name:
    st.write('Add some text to be search. If you are searching for first or last author please use the following format "Durh S"')

# searchButton = st.button('Search', 'search')

if name:
    df = getParsedArticlesPeriod(name, year_to, year_from, term)
    if df is None:
        st.write('No pubmed publications from', name, ' seqarched by ', term, 'between ', year_from, ' – ', year_to)
        st.stop()
    else:
        resultsORrg = pipeline(df, db, tfidf, X)
try:    
    results = resultsORrg.copy()
    sugestions = filter_scores(results)
    grouped_sugestions = grouped_sugestions(sugestions)
    selected_refs =[]
    for table in grouped_sugestions:
        result = pd.DataFrame(table[1])
        st.header(result.iloc[0]['Title'])
        # authors = '; '.join([x['name'] for x in result.iloc[0]['Authors']])
        st.subheader(result.iloc[0]['Published'])
        st.subheader(result.iloc[0]['Authors'])
        # st.table(result)
        # result = result[['score', 'title_x', 'instrument', 'mendeley-tags']]
        # columns = ['Score', 'Mendeley Paper', 'Instrument', 'Mendeley Keys']
        # resul.columns = columns
    #     [ 'queryID', 'dbID', 'Score', 
    # 'DB Paper', 'DB abstract', 'Instrument', 'Keywords DB', 'mendeley-tags', 'Journal DB', 'doi DB', 'Authors DB', 
    # 'Title', 'Authors',  'Abstract',  'Keys', 'Published']
        st.table(result[['Score', 'DB Paper', 'Instrument', 'Keywords DB', 'Keys']].sort_values('Score', ascending=False).iloc[:topN])
    exp_reference = st.button('Export Reference')
    if exp_reference:
        references = sugestions.groupby('dbID').count().sort_values('queryID', ascending=False).index
        # [['author', 'title','journal', 'doi']]
        # references.fillna('', inplace=True)# st.write(references)
        for i in range(0,len(references)):
            row = db.iloc[references[i]][['author', 'title', 'journal', 'doi', 'year', 'url']]
            st.write(f"{i+1}. {row['author']}; \"{row['title']}\"; {row['journal']}; {row['year']}; {row['doi']}; {row['url']}")

except:
    st.stop()
