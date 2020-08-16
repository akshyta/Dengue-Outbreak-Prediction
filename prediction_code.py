# Group No. 12
# Chinmay 2017274
# Akshyta Katyal 2017216
# Anushika Gupta 2017135
'''
    Reference:- https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
                https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
                https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
                
'''

"""# Import Dataset"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

"""# Visualizing Model

# Function to read the dataset
"""

def GetTrainingData():
    ''' 
        params:- None
        return:- TrainingSet
    '''
    train_features = pd.read_csv('/content/dengue_features_train.csv')
    train_labels = pd.read_csv('/content/dengue_labels_train.csv')
#     print(train_features.isna().sum())

    city_encoder = preprocessing.LabelEncoder()
    city_encoder.fit(train_features['city'].tolist())
    encoded_city = np.array(city_encoder.transform(train_features['city'].tolist()))

    X = train_features[train_features.columns[4:24]].to_numpy()

    # new_X =np.hstack((X, np.atleast_2d(encoded_city).T))
    Y = train_labels['total_cases'].to_numpy()

    # print(np.shape(X))
    # print(np.shape(Y))

    # print(train_features[4:24])
    return train_features[train_features.columns[4:24]], train_labels['total_cases'], X, Y

"""# Impute the missing values in the datasrt"""

def ImputeDataset_Regressor(X, Y):
    Imputer = IterativeImputer(max_iter = 100)
    Impute_Model = Imputer.fit(X,Y)
    ImputedX = Impute_Model.transform(X)

    return Impute_Model, ImputedX

def MeanImputing(X, Y):
    Imputer = SimpleImputer(strategy = "mean")
    Impute_Model = Imputer.fit(X,Y)
    ImputedX = Impute_Model.transform(X)

    return Impute_Model, ImputedX


"""# Load the Test file"""

def GetTest():
    test_features = pd.read_csv('/content/dengue_features_test.csv')
    X = test_features[test_features.columns[4:24]].to_numpy()
    return X,test_features

"""# Preprocess the data
Normalizing to 0 mean and variance 1
"""

def normalize_data(X):
    X_scaled = preprocessing.scale(X)
    return X_scaled

"""# PLot Errors"""

def Plot_TrainingValidationError(x_axis, TrainingError, ValidationError):
    
    plt.figure()
    plt.title("Ttraining Error and Validation Error VS Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.plot(x_axis, TrainingError, label = "TrainingError",  c = "blue")
    plt.plot(x_axis, ValidationError, label = "ValidationError", c = "red")
    plt.legend()
    plt.show()

"""# Custom Predict function"""

def predict(Parameters, X):
    
    PredictedValue = np.dot(X,Parameters)
    
    return PredictedValue

"""# Custom Gradient Descent"""

def GradientDescent(X_train, Y_train, X_val, Y_val, learning_rate = 0.01, iterations = 100):
    
    n = np.shape(X_train)[1]
    m = np.shape(X_train)[0]
    
    X0 = np.reshape(np.ones(m),(m,1))
    X_train = np.concatenate((X0, X_train), axis = 1)
    
    X0 = np.reshape(np.ones(np.shape(X_val)[0]),(np.shape(X_val)[0],1))
    X_val = np.concatenate((X0, X_val), axis = 1)
    
    Y_train = np.reshape(Y_train,(m,1))
    
    TrainingError = []
    ValidationError = []
    Iterations = []

    Parameters = np.reshape(np.ones(n+1),(n+1,1))
    
    
    for i in range(iterations):
        
        PredictedY = predict(Parameters, X_train)

        Parameters = Parameters - learning_rate*np.dot(np.transpose(X_train), PredictedY - Y_train)/(m)

        TrainingError.append(np.sum(np.absolute(predict(Parameters, X_train) - Y_train))/m)
        ValidationError.append(np.sum(np.absolute(predict(Parameters, X_val) - Y_val))/m)
        Iterations.append(i+1)
        
    Plot_TrainingValidationError(Iterations, TrainingError, ValidationError)

"""# Plot results for grid search cv for SVM Model"""

def PlotSVR_DegreeVsMAE(grid_search_model):
    
    grid_search_results = grid_search_model.cv_results_
    print(grid_search_results.keys())
    
    Scores = grid_search_results['mean_test_score']
    Degree = grid_search_results['param_degree']
    
    plt.figure()
    plt.title("Degree vs Mean Cross Validation Score")
    plt.xlabel("Degree")
    plt.ylabel("Mean CV Score")
    plt.plot(Degree, Scores)
    plt.show()

"""# Plot results for grid search cv for ElasticNet Model"""

def PlotElasticNet_L1ratioVsMAE(grid_search_model):
    grid_search_results = grid_search_model.cv_results_
    print(grid_search_results.keys())
    
    Scores = grid_search_results['mean_test_score']
    Degree = grid_search_results['param_l1_ratio']
    
    plt.figure()
    plt.title("Degree vs Mean Cross Validation Score")
    plt.xlabel("L1 ratio")
    plt.ylabel("Mean CV Score")
    plt.plot(Degree, Scores)
    plt.xlim((1.1,2))
    plt.show()

"""# Basiline Models


> Linear regression (without regularization)
> Ridge Regression
> Lasso Regression
"""

def Fit_Baseline(X_train, Y_train, X_test, Y_test):

    Linear_Regressor = LinearRegression()
    Linear_Regressor = Linear_Regressor.fit(X_train,Y_train)

    print("Accuracy with Linear Regression : ")
    evaluation_function(Linear_Regressor, X_test, Y_test)

    Ridge_Regressor = Ridge()
    Ridge_Regressor = Ridge_Regressor.fit(X_train, Y_train)

    print("Accuracy with L1 Regularization : ")
    evaluation_function(Ridge_Regressor, X_test, Y_test)

    Lasso_Regression = Lasso()
    Lasso_Regression = Lasso_Regression.fit(X_train, Y_train)

    print("Accuracy with L2 Regularization : ")
    evaluation_function(Lasso_Regression, X_test, Y_test)

    return Lasso_Regression

"""# Fitting ElasticNet Model"""

def Elastic_Net(X_train,Y_train ,X_test, Y_test):
    
    #[0.0001, 0.001, 0.01, 0.1, 1, 10, 100] = alpha
    
    parametersGrid = {"alpha": [0.01],"l1_ratio": np.array([0.0, 1.0, 0.1, 0.01, 0.03, 0.05, 0.3, 0.5, 0.75, 2])}
    
    elastic_net_model = ElasticNet(max_iter = 1000)
    
    MAE_scorer = make_scorer(mean_absolute_error)
    
    elastic_grid_model = GridSearchCV(elastic_net_model, parametersGrid, scoring = MAE_scorer, cv=3)
    elastic_grid_model.fit(X_train, Y_train)
    
    print("Elastic Net Model:- ")
    print("Best Parameters = " + str(elastic_grid_model.best_params_))
    
    PlotElasticNet_L1ratioVsMAE(elastic_grid_model)
    
    ElastricNet_Score = evaluation_function(elastic_grid_model, X_test, Y_test)
    
    return elastic_grid_model, ElastricNet_Score

"""# Fitting Support Vector Regressor"""

def SupportVectorRegressor(X_train, Y_train, X_test, Y_test):
    
    #[0.005,0.05,0.5] = C
    #[0.001,0.3,0.5] = epsilon
    #[1.5,2.5,2,3,1] = degree
    #['poly','linear', 'rbf'] =  kernel
    
    parameters = {'kernel':np.array(['rbf']), 'C':np.array([0.05]), 'degree':[1,2,3,4,5,6,7,8,9,10], 'epsilon':np.array([0.001])}
    Support_Vector_Regressor = SVR(gamma='scale')
    
    # MAE_scorer = make_scorer(mean_absolute_error)
    
    Support_Vector_Regressor = GridSearchCV(Support_Vector_Regressor, parameters, cv=10)
    
    
    Support_Vector_Regressor = Support_Vector_Regressor.fit(X_train,Y_train)
    
    print("Support Vector Regressor Model:- ")
    print("Best Parameters = " + str(Support_Vector_Regressor.best_params_))
    
    Support_Vector_Regressor_Score = evaluation_function(Support_Vector_Regressor, X_test, Y_test)
    
    PlotSVR_DegreeVsMAE(Support_Vector_Regressor)
    
    return Support_Vector_Regressor, Support_Vector_Regressor_Score

"""# Fitting Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor
def randomForestRegressor_model(X,Y, X_test, Y_test):
    regr = RandomForestRegressor(n_estimators = 250, max_depth = 4)
    regr.fit(X, Y)

    RandomForestRegressorscores = evaluation_function(regr, X_test, Y_test)

    return regr, RandomForestRegressorscores

"""# Fitting Decision Tree Regressor"""

from sklearn.tree import DecisionTreeRegressor
def DecisionTreeRegressor_model(X,Y, X_test, Y_test):
    DecisionTreeRegressorModel = DecisionTreeRegressor(criterion = 'mae')
    DecisionTreeRegressorModel.fit(X, Y)

    DecisionTreeRegressorscores = evaluation_function(DecisionTreeRegressorModel, X_test, Y_test)

    return DecisionTreeRegressorModel, DecisionTreeRegressorscores

"""# Bayesian Ridge Regression"""

def BayesianRedge_Model(X,Y,X_test,Y_test):
    BayesianRidgeModel = BayesianRidge(n_iter = 1000)
    BayesianRidgeModel = BayesianRidgeModel.fit(X, Y)

    
    BayesianRidgeModelscores = evaluation_function(BayesianRidgeModel, X_test, Y_test)

    return BayesianRidgeModel, BayesianRidgeModelscores

"""# MLP"""

def MLP_Model(X,Y,X_test,Y_test):
    MLP = MLPRegressor((20,16,8,4,2,1), alpha = 0.001, activation = 'relu', batch_size = 50)
    MLP = MLP.fit(X, Y)

    # Training Error
    print("~~~~~~~~~~~~~Training Error~~~~~~~~~~~~~~~~~~")
    MLP_score = evaluation_function(MLP, X, Y)

    # Test Error
    print("~~~~~~~~~~~~~Test Error~~~~~~~~~~~~~~~~~~")
    MLP_score = evaluation_function(MLP, X_test, Y_test)

    return MLP, MLP_score

"""MLP Classifier"""

def MLP_Classifier(X,Y,Xtest,Ytest):
    MLP = MLPClassifier((12,8,5,3,1), alpha = 0.001, activation = 'relu', batch_size = 50)
    MLP = MLP.fit(X, Y)

    # Training Error
    print("~~~~~~~~~~~~~Training Error~~~~~~~~~~~~~~~~~~")
    MLP_score = evaluation_function(MLP, X, Y)

    # Test Error
    print("~~~~~~~~~~~~~Test Error~~~~~~~~~~~~~~~~~~")
    MLP_score = evaluation_function(MLP, X_test, Y_test)

    return MLP, MLP_score

"""GradientBoosting"""

def GradientBoosting(X,Y,Xtest,Ytest):
    GradientBoostingModel = GradientBoostingRegressor(loss = 'lad', n_estimators=250, criterion = "mae")
    GradientBoostingModel = GradientBoostingModel.fit(X, Y)

    # Training Error
    print("~~~~~~~~~~~~~Training Error~~~~~~~~~~~~~~~~~~")
    GradientBoostingModel_score = evaluation_function(GradientBoostingModel, X, Y)

    # Test Error
    print("~~~~~~~~~~~~~Test Error~~~~~~~~~~~~~~~~~~")
    GradientBoostingModel_score = evaluation_function(GradientBoostingModel, Xtest, Ytest)

    return GradientBoostingModel, GradientBoostingModel_score

"""# Model Selector
Chooses a model that gives least error on the validation set
"""

def Model_Selection(X_train, Y_train, X_test, Y_test):
    
    ElasticNet_Model, ElasticNet_Error = Elastic_Net(X_train, Y_train, X_test, Y_test)
    
    SVR_Model, SVR_Error = SupportVectorRegressor(X_train, Y_train, X_test, Y_test)

    RandomForestModel, RandomForestRegressor_scores = randomForestRegressor_model(X_train,Y_train, X_test, Y_test)

    BayesianRidgeModel, BayesianRidgeModelscores = BayesianRedge_Model(X_train, Y_train, X_test, Y_test)

    DecisionTreeRegressorModel, DecisionTreeRegressorscores = DecisionTreeRegressor_model(X_train, Y_train, X_test, Y_test)

    MLPRegressorModel, MLPRegressorModel_score = MLP_Model(X_train, Y_train, X_test, Y_test)

    MLPClassifierModel, MLPClassifierrModel_score = MLP_Model(X_train, Y_train, X_test, Y_test)

    GradientBoostingModel, GradientBoostingModel_score = GradientBoosting(X_train, Y_train, X_test, Y_test)


    return GradientBoostingModel

"""# Evaluation on Different Evaluation Matrices"""

def evaluation_function(model,X_test,Y_test):
    Y_pred = model.predict(X_test)
    Y_pred = result_normalizer(Y_pred)
    print('r2 score : ',r2_score(Y_test, Y_pred)) 
    print('Max error : ',max_error(Y_test, Y_pred))
    print('Mean Suared Error :  ',mean_squared_error(Y_test,Y_pred))
    print('Mean Absolute error : ', mean_absolute_error(Y_test,Y_pred))
    print("\n\n")
    return mean_absolute_error(Y_test,Y_pred)

"""# Predition on test set"""

def PredictForTest(Impute_Model, Model, selector):

    X_df = pd.read_csv("/content/dengue_features_test.csv")
    Identifiers =  X_df[X_df.columns[0:3]]
    X_test = X_df[X_df.columns[4:24]]
    Imputed_X_test = Impute_Model.transform(X_test)

    X_test = normalize_data(Imputed_X_test)
    X_test = selector.transform(X_test)
    Y_res = Model.predict(X_test)
    Y_res = result_normalizer(Y_res)

    Identifiers['total_cases'] = pd.Series(Y_res)
    Identifiers.to_csv("/content/submission.csv")

"""# Normalizing Results
Assigning negatives values to 0.
"""

def result_normalizer(res):
    output = res
    output[output < 0] = 0
    output = np.rint(output)
    output = output.astype('int')
    return output

"""# Data Visualization

Heatmap
"""

def MakeHeatMap(df_data, df_labels, X, Y):

    i = 0

    for column in df_data.columns:
        df_data[column] = pd.Series(X[:,i])
        i += 1

    df_data["MEDV"] = pd.Series(Y)

    plt.figure(figsize=(20,20))
    cor = df_data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    #Correlation with output variable
    cor_target = abs(cor["MEDV"])

    # Step-1 ------ Select features 
    Features = cor_target[cor_target > 0.1]

    # print(df_data[[Features[0],Features[1]]].corr()[Features[1]][0])

    RepeatedFeatures = []

    for k1 in range(len(Features)):
        for k2 in range(k1+1,len(Features)):

            if Features[k1] != "MEDV" and Features[k2] != "MEDV":

                if df_data[[Features[k1],Features[k2]]].corr()[Features[k1]][1] > 0.5:
                    if df_data[[Features[k2],"MEDV"]].corr()[Features[k2]][1] > df_data[[Features[k1],"MEDV"]].corr()[Features[k1]][1]:
                        if Features[k1] not in RepeatedFeatures:
                            RepeatedFeatures.append(Features[k1])
                    else:
                        if Features[k2] not in RepeatedFeatures:
                            RepeatedFeatures.append(Features[k2])
    Features = Features.tolist()
    # print(Features)
    # print(RepeatedFeatures)
    for i in range(len(RepeatedFeatures)-1):
        if RepeatedFeatures[i] in Features:
            Features.remove(RepeatedFeatures[i])

    X = []
    for f in Features:
        if f == "MEDV":
            continue
        X.append(df_data[f].tolist())

    X = np.array(X)
    X = X.T
    Y = np.array(Y)

    return X, Y

"""# Outlier Removal Techniques

Isolation Forest Technique
"""

from sklearn.ensemble import IsolationForest
def IsolationForestFunction(X,Y):
    clf = IsolationForest(warm_start=True)
    pred = clf.fit_predict(X)
    deleted = []
    for i in range(len(pred)):
        if pred[i] < 0:
            deleted.append(i)
    X_new = np.delete(X,deleted,0)
    Y_new = np.delete(Y,deleted)
    return X_new,Y_new, clf

"""Local Outlier Factor"""

from sklearn.neighbors import LocalOutlierFactor
def LocalOutlierFactorFunction(X,Y):
    clf = LocalOutlierFactor()
    pred = clf.fit_predict(X)
    deleted = []
    for i in range(len(pred)):
        if pred[i] < 0:
            deleted.append(i)
    X_new = np.delete(X,deleted,0)
    Y_new = np.delete(Y,deleted)
    return X_new,Y_new, clf

"""One Class SVM"""

from sklearn.svm import OneClassSVM
def OneClassSVMFunction(X,Y):
    clf = OneClassSVM()
    pred = clf.fit_predict(X)
    deleted = []
    for i in range(len(pred)):
        if pred[i] < 0:
            deleted.append(i)
    X_new = np.delete(X,deleted,0)
    Y_new = np.delete(Y,deleted)
    return X_new,Y_new, clf

"""# Main Function"""

def main():

    # Load Dataset    
    df_data, df_label, X_train, Y = GetTrainingData()
    
    # Impute the Dataset
    Impute_Model, ImputedX_train = MeanImputing(X_train, Y)

    # Normalize the Dataset
    X = normalize_data(ImputedX_train)

    # Split the training set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state = 0)

    # Run Model Selector
    Model = Model_Selection(X_train, Y_train, X_val,Y_val)

    #~~~~~~~~~~~ Visualize the Dataset and experimenting on different data selection techniques ~~~~~~~~~~~~~~~
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # A = Correaltion and Heatmaps
    X, Y = MakeHeatMap(df_data, df_labels, X, Y)

    print(X.shape)
    print(Y.shape)

    # Split the training set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state = 0)

    # Run Model Selector
    Model = Model_Selection(X_train, Y_train, X_val,Y_val)



    # B = Recursive Feature extraction

    X,Y, IsolationForestRemovalClassifier = IsolationForestFunction(X,Y)

    estimator = LogisticRegression()
    estimator = SVR(kernel = 'linear')
    selector = RFECV(estimator, step=1, min_features_to_select = 12, cv = 5)
    # selector = RFE(estimator, 12, step=1)
    selector = selector.fit(X, Y)
    X = selector.transform(X)

    # Split the training set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state = 0)


    # Run Model Selector
    Model = Model_Selection(X, Y, X_val,Y_val)
    Model = Fit_Baseline(X_train, Y_train, X_val, Y_val)
    
    Select K best features

    selector = SelectKBest(f_regression, k=12)
    selector = selector.fit(X, Y)
    X = selector.transform(X)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # Run Model Selector
    Model = Model_Selection(X_train, Y_train, X_val,Y_val)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~End~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # PredictForTest(Impute_Model, Model, selector)

    # Fit the baseline
    # Fit_Baseline(X_train, Y_train, X_val, Y_val)
    # Custom Gradient Descent
    # GradientDescent(X_train, Y_train, X_val, Y_val)
    
#     print(Model)
    # res = Model.predict"(X_val)
    # print((res))result_normalizer
#     X_test,test_df =GetTest()
#     print(X_test)
#     PredictForTest(Impute_Model, Model,X_test,test_df)

if __name__ == '__main__':
    main()

#