import sys
from PyQt5 import QtCore, QtGui, uic, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
qtCreatorFile = "MyApp.ui" # Enter file here.
import glob
import csv
import os
import numpy as np
from queue import LifoQueue 
import nltk
from nltk.stem import WordNetLemmatizer 
from scipy import spatial
import math
 
lemmatizer = WordNetLemmatizer()

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
from PyQt5.QtGui import QPalette, QColor


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.SetUpMyBackend()
        self.ClearAllScreen()

    def SetUpMyBackend(self):
        self.pushButtonSearch.clicked.connect(self.Search)

    def ClearAllScreen(self):
        self.pushButtonClear.clicked.connect(self.clearScreen)
    
    def clearScreen(self):
        print("in clear screen function")
        self.plainTextShowResult.setPlainText("")
        self.lineEditCutoff.setText("")
        self.lineEditQuery.setText("")
        self.labelLength.setText("No documents retreived")
        

    #this function is invoked by search button
    def Search(self):
        #taking the query from the text box
        query=self.lineEditQuery.text()

        #taking the cutoff from the text box
        if(self.lineEditCutoff.text()!=""):
            cutoff=float(self.lineEditCutoff.text())
            self.CallallFunctions(query,cutoff)
        else:
            self.CallallFunctions(query)  

    def CallallFunctions(self, query,cutoff=0.005):
        
        #initialize empty results
        self.plainTextShowResult.setPlainText("")
        self.labelLength.setText("No documents retreived")

        #print query and cutoff in output container
        self.plainTextShowResult.appendPlainText("Entered query is : "+query+"\n")
        self.plainTextShowResult.appendPlainText("CutOff value is : "+str(cutoff)+"\n")
    

        #Start query preprocessing
        # removing punctuations from query
        query=query.replace(".","").replace("n't"," not").replace("]"," ").replace("["," ").replace(","," ").replace(")","").replace("?","").replace("/","").replace("(","").split()

        #stemming the query
        stemmedQuery=[lemmatizer.lemmatize(x.lower()) for x in query]

        #remove stop words from query
        QueryWithoutStoplist = [x for x in stemmedQuery if x not in swl]
        print(QueryWithoutStoplist)

        #initialize tfidf_dict{} and call function for doc and query tfidf calculation
        tfidf_dict={}
        tfidf_dict=calculateTFIDFDocsAndQuery(pindex_table,doclist,QueryWithoutStoplist)

        #make doc and query vector having tfidf of words
        docVec, queryVec = DocAndQueryVector(tfidf_dict,doclist)

        #find out cosine similarity b/w query and each doc
        cosineSimilarity(self, docVec, queryVec, cutoff)

        return


#taking input from stop word list file

fObj=open('StopwordList.txt','r')
SwContent=fObj.readlines()
swlist = [x.replace("\n","").replace(" ","") for x in SwContent]
swl = [x for x in swlist if x!=""]           # stop word list stored in 'swl'

ldict={} #lowcase doc
ldict1={} #without stop list

# taking input from file and preprocessing

list=glob.glob('Speeches/*')
doclist=[]

for x in sorted(list):
#     print(x)
    f=open(x,'r')
    
    # read file and removing punctuations
    fullfile=f.read().replace(".","").replace("n't"," not").replace("'","").replace("]"," ").replace("[","").replace(","," ").replace("?","").replace("\n"," ").replace("-"," ").replace(":"," ").replace("$"," ").split() 

    # stemming and lower case conversion
    lowCasedoc=[lemmatizer.lemmatize(x.lower()) for x in fullfile]
    
    #removing stop words
    withoutstoplist = [x for x in lowCasedoc if x not in swl]
    
    #trimming the file name and removing redundant '.txt'
    p=os.path.basename(x)
    p=p.split('.')[0]
    ldict[p]=lowCasedoc
    ldict1[p]=withoutstoplist
    doclist.append(p)  #maintaining a list of documents


# index creation
term_count_in_each_doc={}
pindex_table={}
wfreq={}
#iterating through docs
for key in ldict1.keys():

    TC=0       # maintain the total count of the terms and can tell their index during iteration

    #iterating through words in each doc
    for word in ldict1[key]:
        TC+=1
        if word in swl:  # entertaining the presence of stop words in the file (increment index without doing anything)
            continue
        if word not in pindex_table:
            pindex_table[word]={}
            pindex_table[word][key]=[]
            pindex_table[word][key].append(TC) 
        else:
            if key not in pindex_table[word]:
                pindex_table[word][key]=[]
            pindex_table[word][key].append(TC)
            
    term_count_in_each_doc[key]=TC     #total no of words in a doc (including stop words) for TTf


def calculateTFIDFDocsAndQuery(pindex_table, doclist,query):
    tfidf_dict={}
    # w = csv.writer(open("TFIDF.csv", "w"))
#     idf=log(N/no of documents with the term)
    f = open("TFIDFdict_Output.txt","w")
 
    for word in pindex_table.keys():

        tfidf_dict[word]={}
        dfreq=0
        #doc freq
        dfreq=len(pindex_table[word])

        #idf
        idf=math.log(56/(dfreq))

        tfidf_dict[word]=[]   # initializing 'tfidf values in each doc' list against the word
        for i in range(len(doclist)):
            x='speech_{}'.format(i)
            if(x in pindex_table[word].keys()):  #if term exist in document
                tfidf_dict[word].append((len(pindex_table[word][x])*idf))  #append tfidf of word in the respective document
            else:
                tfidf_dict[word].append(0)      #if term doesn't exist in document
            # w.writerow([word, x,tfidf_dict[word]])
        if word in query:
            tfidf_dict[word].append(round(((query.count(word))*idf),4)) #tfidf of the query at 56th index of each word list
        else:
            tfidf_dict[word].append(0)
            
    # print(tfidf_dict)
    f.write( str(tfidf_dict) )
    f.close()
                
    return tfidf_dict
    
#function to get document and query 
def DocAndQueryVector(tfidf_dict,doclist):
#     print(tfidf_dict.keys())

    docVec={} # a vector of all words will be maintained against the document key  (total 56 keys and each key will have a list against it)
    
    #iterating through docs
    for docNo in range(len(doclist)):
        
        x='speech_{}'.format(docNo)
        docVec[x]=[]
        
        for word in tfidf_dict.keys():
            docVec[x].append(tfidf_dict[word][docNo])

    queryVec=[]
    
    for word in tfidf_dict.keys():
        queryVec.append(tfidf_dict[word][56])     
            
    return docVec, queryVec


#cosine similarity function

def cosineSimilarity(ui, docVec, queryVec, c=0.005):
    
    cosineVector={}
    cosineList=[]
    for i in docVec.keys(): 
        result = round(1 - spatial.distance.cosine(docVec[i], queryVec),3)
        cosineVector[i]=result
        cosineList.append(result)
        
    sortedDocIds=sorted(range(len(cosineList)), key=cosineList.__getitem__, reverse=True) # get sorted document ids
    x_axisList=[]
    y_axisList=[]
    w = csv.writer(open("CosineSimilarities_Output.csv", "w"))
    #filtering out the docs wrt cutOff and print them on the output screen
    counter=0
    for i in sortedDocIds:
        if(cosineList[i] > c):
            counter=counter+1
            if i<=9:
                ui.plainTextShowResult.appendPlainText("Document "+str(i)+"    ---->    "+str(cosineList[i]))
            elif 9<i<=99:
                ui.plainTextShowResult.appendPlainText("Document "+str(i)+"  ---->    "+str(cosineList[i]))
            x_axisList.append(str(i))
            y_axisList.append(cosineList[i])
            
        w.writerow(["Doc_{}".format(i), cosineList[i]])

    plt.plot(x_axisList, y_axisList, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12) 
    plt.xlabel('Document ID')
    plt.ylabel('Score')
    plt.savefig('TFIDF_ScoreResults.png')
    plt.show()

    #if no relevent documents are found        
    if(counter==0):
        ui.plainTextShowResult.appendPlainText("No Documents found !")

    ui.labelLength.setText("Retreived Documents: "+str(counter))

    
    # print(cosineSimilarity)   
    return

# pyQT palette     
if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    app.setStyle("Fusion")

    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    window.show()
    sys.exit(app.exec_())