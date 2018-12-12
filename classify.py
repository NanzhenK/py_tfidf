# -*- coding: utf-8 -*-
class Classify:

    def __init__(self,train_x,train_y,classifier):
        classifiers = {'NB':self.naive_bayes_classifier,   
				  'KNN':self.knn_classifier,  
				   'LR':self.logistic_regression_classifier,  
				   'RF':self.random_forest_classifier,  
				   'DT':self.decision_tree_classifier ,
				   'SVM':self.svm_classifier,  
				   'SVMCV':self.svm_cross_validation,  
				   'GBDT':self.gradient_boosting_classifier  
        }  
        self.model = classifiers[classifier](train_x, train_y)
    
    # Multinomial Naive Bayes Classifier  
    def naive_bayes_classifier(train_x, train_y):  
        from sklearn.naive_bayes import MultinomialNB  
        model = MultinomialNB(alpha=0.01)  
        model.fit(train_x, train_y)  
        return model  
    # KNN Classifier  
    def knn_classifier(train_x, train_y):  
        from sklearn.neighbors import KNeighborsClassifier  
        model = KNeighborsClassifier()  
        model.fit(train_x, train_y)  
        return model  
 
    # Logistic Regression Classifier  
    def logistic_regression_classifier(train_x, train_y):  
        from sklearn.linear_model import LogisticRegression  
        model = LogisticRegression(penalty='l2')  
        model.fit(train_x, train_y)  
        return model  
    
    # Random Forest Classifier  
    def random_forest_classifier(train_x, train_y):  
        from sklearn.ensemble import RandomForestClassifier  
        model = RandomForestClassifier(n_estimators=8)  
        model.fit(train_x, train_y)  
        return model  
    
    # Decision Tree Classifier 
    def decision_tree_classifier(train_x, train_y):  
        from sklearn import tree  
        model = tree.DecisionTreeClassifier()  
        model.fit(train_x, train_y)  
        return model  
    
    # GBDT(Gradient Boosting Decision Tree) Classifier  
    def gradient_boosting_classifier(train_x, train_y):  
        from sklearn.ensemble import GradientBoostingClassifier  
        model = GradientBoostingClassifier(n_estimators=200)  
        model.fit(train_x, train_y)  
        return model  
    
    # SVM Classifier  
    def svm_classifier(train_x, train_y):  
        from sklearn.svm import SVC  
        model = SVC(kernel='rbf', probability=True)  
        model.fit(train_x, train_y)  
        return model  
  
    # SVM Classifier using cross validation  
    def svm_cross_validation(train_x, train_y):  
        from sklearn.grid_search import GridSearchCV  
        from sklearn.svm import SVC  
        model = SVC(kernel='rbf', probability=True)  
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
        grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
        grid_search.fit(train_x, train_y)  
        best_parameters = grid_search.best_estimator_.get_params()  
        for para, val in best_parameters.items():  
            print(para, val)  
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
        model.fit(train_x, train_y)  
        return model  
        
    def predict(self,test_x):
        self.predict = self.model.predict(test_x)
        return self.predict
        
    def score(self,test_y):
        from sklearn import metrics
        precision_macro = metrics.precision_score(test_y, self.predict, average='macro')
        recall_macro = metrics.recall_score(test_y, self.predict, average='macro')  
        precision_micro = metrics.precision_score(test_y, self.predict, average='micro')  
        recall_micro = metrics.recall_score(test_y, self.predict, average='micro')  
        print (' & auto &precision_micro: %.2f%% &precision_macro: %.2f%% & recall_micro: %.2f%% &recall_macro: %.2f%% \\ ' % ( 
                                                                                                                  100 * precision_micro, 
                                                                                                                  100 * precision_macro,
                                                                                                                  100 * recall_micro,
                                                                                                                  100 * recall_macro)  )
