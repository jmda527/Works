import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.mixture import GaussianMixture



class Ternary_contour:
    '''
    ************************example*************************
    list_axis = ['age','trestbps','chol']
    test = Ternary_contour(
                datax,model_jmda,list_axis,
                resolution=20,predict_type='classification',
                output_class=1
                )
    test.plot(title='ternary plot')


    ***********************parameter************************
    df_x : dataframe , data for predicting
    model : sklearn model , model for predicting
    axis : list , list of features you want to plot
    resolution : int , resolution of grid(default=20)
    predict_type : str , type of model , 'regression' or 'classification'
    output_class : int , only classification model needs , target of output


    '''
    y_score = []
    y_score_sum = []
    tri_grid = []
    
    def __init__(self,df_x,model,axis,resolution=20,predict_type='regression',output_class=0):
        self.df_x = df_x
        self.model = model
        self.axis = axis
        self.resolution = resolution
        self.predict_type = predict_type
        self.output_class = output_class

        
    #make grid
    def get_grid(self):
        self.tri_grid = []
        for i in range(self.resolution+1):
            for j in range(self.resolution+1):
                for k in range(self.resolution+1):
                    if i+j+k == self.resolution:
                        self.tri_grid.append([i,j,k])
        self.tri_grid = np.array(self.tri_grid)

        
    #partial dependence with one axis fixed
    def predict_data_dependence(self,fix):
        self.y_score = []
        for i in self.tri_grid:
            data_grid = self.df_x.copy()
            data_base = self.df_x.copy()
                
            for a in range(3):
                data_grid[self.axis[a]] = self.df_x[self.axis[a]].min() + (self.df_x[self.axis[a]].max()-self.df_x[self.axis[a]].min())*i[a]/self.resolution
                if a == fix:
                    data_base[self.axis[a]] = self.df_x[self.axis[a]].min()
                else:
                    data_base[self.axis[a]] = self.df_x[self.axis[a]].min() + (self.df_x[self.axis[a]].max()-self.df_x[self.axis[a]].min())*i[a]/self.resolution
           
            if self.predict_type == 'regression':
                y_pred = self.model.predict(data_grid)
                y_base = self.model.predict(data_base)
            elif self.predict_type == 'classification':   
                y_pred = self.model.predict_proba(data_grid)[:,self.output_class]
                y_base = self.model.predict_proba(data_base)[:,self.output_class]

            y_pred_n = (y_pred/y_base-1)
            self.y_score.append(GaussianMixture().fit(y_pred_n.reshape(-1,1)).means_[0][0])
     
    
    #sum of partial dependence(3)
    def predict_data_mean(self):
        self.y_score_sum = []
        self.get_grid()
        for i in range(3):
            self.predict_data_dependence(i)
            self.y_score_sum.append(self.y_score)
        self.y_score_sum = np.array(self.y_score_sum).sum(axis=0)
    
    
    #plot ternary figure
    def plot(self,title):
        self.predict_data_mean()
        fig = ff.create_ternary_contour(
                                np.array([self.tri_grid[:,i] for i in range(3)])/self.resolution,
                                self.y_score_sum,
                                pole_labels=self.axis,
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Rainbow',
                                showmarkers=True,
                                showscale=True,
                                title=title
        )
        fig.show()



