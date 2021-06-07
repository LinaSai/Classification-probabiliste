# coding: utf-8
import math
import utils 
from functools import reduce
import operator
import collections.abc
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt

#Question 1 : ##################################################################################################

def getPrior(df):
    """
    Calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de confiance 
    à 95% pour l'estimation de cette probabilité.
    Aruguments :
    df : dataframe
    """
    mean = df["target"].mean()
    sd = df["target"].std()
    count_row=len(df.index)
    confidence_coefficient=1.96 #pour trouver cette valeur nous avons calculé la valeur de alpha/2 = 0.95/2=0.475
    # nous avons ensuite cherche cette valeur dans la table de la loi de Z(coefficient de confidence ) 
    # et avons somme la valeur dans la ligne et colonne : 1.9 +0.06 

    d = dict()
    d['estimation'] = mean
    d['min5pourcent'] = mean - (confidence_coefficient * (sd/math.sqrt(count_row)))
    d['max5pourcent'] = mean + (confidence_coefficient * (sd/math.sqrt(count_row)))
    print("Résultat :",d)
    return d

#Question 2 : ##################################################################################################
#Question 2.a : 
class APrioriClassifier(utils.AbstractClassifier):
    def ___init__(self):
        pass

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        return 1

#Question 2.b : 

    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        d = dict()
        d['VP'] = 0
        d['VN'] = 0
        d['FP'] = 0
        d['FN'] = 0
        d['précision'] = 0
        d['rappel'] = 0

        i=0
        for t in df.itertuples():
            dic=t._asdict()
            target_value = dic['target']
            attrs = utils.getNthDict(df,i)
            predicted_value = self.estimClass(attrs)

            if (predicted_value==target_value):
                if (target_value==1):
                    d['VP']+=1
                else :
                     d['VN']+=1  
            else :
                if (target_value==1):
                    d['FN']+=1
                else :
                     d['FP']+=1  

            i+=1
        d['précision'] = d['VP']/(d['VP']+d['FP'])
        d['rappel'] = d['VP']/(d['VP']+d['FN'])
        #print("Résultat",d)
        return d
        
#Question 3 : ##################################################################################################
#Question 3.a :
#  
def P2D_l(df,attr):
    """
    Calcule dans le dataframe la probabilité P(attr|target) sous la forme d'un dictionnaire asssociant 
    à la valeur t un dictionnaire associant à la valeur a la probabilité P(attr=a|target=t)
    
    Arguments:
    df: dataframe
    attr: un attribut ( qui correspond a une colonne du dataframe)
    """

    res = dict()
    values_of_target_distinct=df['target'].unique()
    values_of_attribute_distinct=df[attr].unique()
    N_target= df.groupby('target')[attr].count()  

    for target in values_of_target_distinct: # on initilialise le dictionnaire qui sera de la forme {0{value1 : proba 1,..},1{value1 : proba 1}}
        dic_tmp = dict()
        for val_attr in values_of_attribute_distinct:
            dic_tmp[val_attr] = 0
        res[target] = dic_tmp

    
    for t in df.itertuples():
        dictio = t._asdict()
        target = dictio['target'] 
        attribut = dictio[attr] 
        res[target][attribut] += 1 #on compte le nombre de patients ayant P(attr=a|target=t)
    
    
    for target in res.keys():
        for val_attribut in res[target].keys():
            res[target][val_attribut] /= N_target[target] # on divise pour la valeur de t, par P(target=t) pour avoir les probabilites

    return res


def P2D_p(df,attr):
    """
    Calcule dans le dataframe la probabilité P(target|attr) sous la forme d'un dictionnaire asssociant 
    à la valeur t un dictionnaire associant à la valeur a la probabilité P(target=t|attr=a)
    
    Arguments:
    df: dataframe
    attr: un attribut ( qui correspond a une colonne du dataframe)
    """
    #meme principe que P2D_l mais ici calcule P(target|attr)

    values_of_target_distinct=df['target'].unique()
    values_of_attribute_distinct=df[attr].unique()
    N_attribute= df.groupby(attr)['target'].count() 

    res = dict()
    for target in values_of_attribute_distinct:
        dic_tmp = dict()
        for val_attr in values_of_target_distinct:
            dic_tmp[val_attr] = 0
        res[target] = dic_tmp


    for t in df.itertuples():
        dictio = t._asdict()
        target = dictio['target'] 
        attribut = dictio[attr] 
        res[attribut][target] += 1
    
    
    for val_attribut  in res.keys():
        for target in res[val_attribut].keys():
            res[val_attribut][target] /= N_attribute[val_attribut]

    return res


#Question 3.b : 
class ML2DClassifier(APrioriClassifier):
    
    def __init__(self,df,attr):
        """
        Initiliase un classifieur qui utilise le maximum de vraisemblance 
        pour estimer la classe d'un individu.

        """
        self.table_P2Dl=P2D_l(df,attr)
        self.attr=attr
        self.df=df


    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        value_attribute=attrs[self.attr]  
        return ( self.table_P2Dl[1][value_attribute] > self.table_P2Dl[0][value_attribute])
            

#Question 3.c : 

class MAP2DClassifier(APrioriClassifier):

    def __init__(self,df,attr):
        """
        Initiliase un classifieur qui utilise le maximum a posteriori 
        pour estimer la classe d'un individu.

        """
        self.table_P2Dp=P2D_p(df,attr)
        self.attr=attr
        self.df=df


    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        value_attribute=attrs[self.attr]
        return ( self.table_P2Dp[value_attribute][1] > self.table_P2Dp[value_attribute][0])


            
#Question 4 : ##################################################################################################
#Question 4.1 : 

def nbParams(df,liste=None):
    """
    calcule la taille mémoire des tables P(target|attr1,attr2,..)

    Arguments:
    df : dataframe
    liste: liste d'attributs

    """
    if(liste==None): #si on ne donne aucune liste, on prend en parametre toute la liste des attributs
        liste=list(df)

    length = len(liste)
    res=8
    for l in liste :
        res *= len(df[l].unique()) 
    print ("{} variable(s) : {} octets".format(length , res))
    


#Question 4.2 : 
def nbParamsIndep(df):
    """
    calcule la taille mémoire des tables P(target|attr1,attr2,..) , EN SUPPOSANT L'INDEPENDANCE DES ATTRIBUTS

    Arguments:
    df : dataframe
    liste: liste d'attributs

    """
    array=list(df)
    length = len(array)
    res=0
    for a in array :
        res += len(df[a].unique())

    print ("{} variable(s) : {} octets".format(length , res*8))
        

#Question 5 : ##################################################################################################
#Question 5.3 : 
        
def drawNaiveBayes(df,target):
    """
    Dessine le graphe des indépendances conditionnelles des attributs

    Argument:
    df : dataframe
    target : attribut target (parent de tous les autres attributs)

    """
    array=list(df)
    string=""
    for a in array:
        if(a != target): #Pour eviter la boucle de target vers lui meme 
            string+=("{}->{};".format(target , a))
    return utils.drawGraph(string)

def nbParamsNaiveBayes(df,target_attribute,array=None):
    """
    calcule la taille mémoire des tables P(target|attr1,attr2,..) , EN FAISANT L'HYPOTHESE DU NAIVE BAYES

    Arguments:
    df : dataframe
    liste: liste d'attributs

    """
    res=0
    if(array==None):
        array=list(df)
    if(array):
        for a in array:
            if(a!=target_attribute):
                res+=len(df[target_attribute].unique())*len(df[a].unique())
            else:
                res+=len(df[target_attribute].unique())*1 #car P(target|target)=P(target)
    else:
        res=len(df[target_attribute].unique())
    length=len(array)
    print("{} variable(s) : {} octets".format(length , res*8))



#Question 5.4 : 


class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        """ Initiliase le classifier MLNaiveBayes (qui utilise le maximum de la vraisemblance ), 
        calcule les parametres du NaiveBayes.

        Arguments:
        df : le dataFrame
        """
        self.df = df 
        self.array_attributes=list(df.columns)
        self.array_attributes.remove("target")
        self.contingence=dict() 
        
        for attribut in self.array_attributes:
            self.contingence[attribut]= P2D_l(df, attribut)
        
             
    def estimProbas(self,attrs):
        """
        Rend un dictionnaire des probabilites pour target=0 ainsi que target=1 
        en se basant sur la formule : P(attr1,attr2,attr3,...|target) = P(attr1|target) * P(attr2|target) * P(attr3|target),...
  
        Arguments:
        attrs -- un dictionnaire d'attributs (qui represente un patient)
        """  
        d= dict() 
        d[0]=1
        d[1]=1
        
        for attr in self.contingence:
        
            dic_tmp=self.contingence[attr] 
            value_attribute_of_patient=attrs[attr] 

            if value_attribute_of_patient not in dic_tmp[0] or value_attribute_of_patient not in dic_tmp[1] :
                d[0]=0 
                d[1]=0
                return d

                
            d[0] *=  dic_tmp[0][value_attribute_of_patient]
            d[1] *=  dic_tmp[1][value_attribute_of_patient]
                
        return d
                

    def estimClass(self,attrs) :
        """
        Estime la classe 0 ou 1, fait appel
        a estimProbas pr l'estimation
        
        Arguments :
        attrs : dictionnaire d'attributs (qui represente un patient)

        """
        valeurs = self.estimProbas(attrs)
        return int(valeurs[1]>=valeurs[0])


class MAPNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df):
        """ 
        Initiliase le classifier MAPNaiveBayes (qui utilise le maximum a posteriori ),
        calcule les parametres du NaiveBayes.

        Arguments:
        df : le dataFrame
        """
        self.df=df
        self.array_attributes=list(df.columns)
        self.array_attributes.remove("target")

        self.contingence=dict()
        for attribut in self.array_attributes:
            self.contingence[attribut] = P2D_l(df,attribut) #c'est la fonction qui donne P(target|attr)
      

        
    def estimProbas(self,attrs):
        """ Rend un dictionnaire des probabilites pour target=0 ainsi que target=1 
        en se basant sur la formule : P(target|attr1,attr2,attr3,...)= P(target)* [P(attr1|target)*P(attr2|target)*P(attr3|target),...] / [P(attr1,attr2,attr3,...)] 
        avec P(attr1,attr2,attr3,...) = P(attr1) * P(attr2) * P(attr3),... car hypothese naive Bayes
  
        Arguments:
        attrs -- un dictionnaire d'attributs (qui represente un patient)
        
        """

        d=dict()
        d[1]=  self.df["target"].mean()    # P(target=1)
        d[0]=1 - d[1] #P(target=0) = 1-P(target=1) 
       
        for attr in self.contingence:
            
            dic_tmp = self.contingence[attr]      
            value_attribute_of_patient= attrs[attr]
                      
            if value_attribute_of_patient not in dic_tmp[0] or value_attribute_of_patient not in dic_tmp[1] :
                d[0]=0 
                d[1]=0
                return d 
                   
            d[0]=d[0] * dic_tmp[0][value_attribute_of_patient]
            d[1]=d[1] * dic_tmp[1][value_attribute_of_patient]
        
        denominator=d[0]+d[1]
        d[0] /= denominator
        d[1] /= denominator
        
        return d
    
    def estimClass(self, attrs):
        """
        Estime la classe 0 ou 1, fait appel
        a estimProbas pr l'estimation
        
        Arguments :
        attrs : dictionnaire d'attributs (qui represente un patient)

        """
        valeurs = self.estimProbas(attrs)
        return int(valeurs[1]>=valeurs[0])
    


#Question 6 : ##################################################################################################


def isIndepFromTarget(df,attr,x):
    """
    Retourne 1 si attr est indépendant de target au seuil de x%.

    Arguments:
    df : dataframe
    attr : attribut
    x : seuil en %
    """
    contingence = pd.crosstab(df[attr],df.target).values 
    g, p, dof, expctd = chi2_contingency(contingence)
    return (p>=x)
          





class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    
    def __init__(self,df,seuil):
        """ 
        Initiliase le classifier ReducedMLNaiveBayes (qui utilise le maximum de vraisemblance) 
        en optimisant avant grace a des
        tests d'independance a la hauteur d'un seuil,
        calcule les parametres du classifieur.

        Arguments:
        df : le dataFrame
        """
        super(ReducedMLNaiveBayesClassifier,self).__init__(df)
        self.seuil=seuil	
        liste_attributes=self.array_attributes

        for attr in liste_attributes:
            attribut_dependance = isIndepFromTarget(df, attr, self.seuil)
            if attribut_dependance : # si l'attribut est independant de target a la hauteur du seuil donne , on le retire
                self.array_attributes.remove(attr)
                self.contingence.pop(attr)
        

    
    def draw(self):
        string = ""
        for a in self.array_attributes:
            string+=("{}->{};".format('target' , a))
        return utils.drawGraph(string)


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):

    def __init__(self, df, seuil):
        """ 
        Initiliase le classifier ReducedMAPNaiveBayes (qui utilise le maximum a posteriori) en optimisant avant grace a des
        tests d'independance a la hauteur d'un seuil,
        calcule les parametres du classifieur.

        Arguments:
        df : le dataFrame
        """
        super(ReducedMAPNaiveBayesClassifier,self).__init__(df)
        self.seuil=seuil	
        liste_attributes=self.array_attributes

        for attr in liste_attributes:
            attribut_dependance = isIndepFromTarget(df, attr, self.seuil)
            if attribut_dependance : # si l'attribut est independant de target a la hauteur du seuil donne , on le retire
                self.array_attributes.remove(attr)
                self.contingence.pop(attr)

    

    def draw(self):
        string = ""
        for a in self.array_attributes:
            string+=("{}->{};".format('target' , a))
        return utils.drawGraph(string)



#Question 7 : ##################################################################################################
#Question 7.2 :  

def mapClassifiers(dic,df):
    """
    Represente graphiquement les classifieurs dans un espace (precision,rappel)

    Arguments
    dic : dictionnaire des classifieurs 
    df : le dataframe
    """
    X_precision=[]
    Y_rappel=[]
    
    for key in dic.keys():
        stat = dic[key].statsOnDF(df) #c'est la fonction statsOnDF qui retourne le dictionnaire avec precision et rappel
        X_precision.append( stat["précision"])
        Y_rappel.append( stat["rappel"])

    fig=plt.figure()
    tool=fig.add_subplot(1,1,1)
    tool.grid(True)
    plt.plot(X_precision,Y_rappel,'x', color='red')

    
    for key in dic.keys():
        tool.annotate(key,(X_precision[int(key)-1],Y_rappel[int(key)-1]))
    plt.show()
    

#Question 8 : ##################################################################################################
#Question 8.1 : 

def MutualInformation(df,X,Y):
    """
    Calcule l'informations mutuelle en utilisant la formule I(X,Y)
    Arguments:
    df : dataframe
    X et Y les deux variables aleatoires qui representes les deux attributs
    """

    proba_table = df.groupby(X)[Y].value_counts() / df.groupby(X)[Y].count()

    #cross=pd.crosstab(df[X],df[Y])
    #cross.apply(lambda r: r/r.sum(), axis=1)
    
                                
    mutual = 0.0
    
    list_values_index = proba_table.index.values.tolist()
    dict_key_unique_values = {}

   
    for x in df[X].unique():
        dict_key_unique_values[x] = []

    for (x,y) in list_values_index:
        dict_key_unique_values[x].append(y)
    
   
    for x in df[X].unique():

        P_x = (df[X].value_counts().div(len(df)))[x]
        
        for y in df[Y].unique():

            if y not in dict_key_unique_values[x]:
                continue

            P_y = (df[Y].value_counts().div(len(df)))[y]
            Px_y = proba_table[x][y] * P_x 
            
            mutual += Px_y * math.log(Px_y/(P_x * P_y) ,2)

    return mutual



