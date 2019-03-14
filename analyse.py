#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import argparse
import csv
import matplotlib.pyplot as plt
import string
import math
from mpl_toolkits.mplot3d import Axes3D
from plotly import tools
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import random
import sklearn
#import sklearn.datasets
from sklearn import datasets, model_selection, metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score


def load_data(input_name):
	'''Loading data from sklearn'''
	names = [name for name in dir(sklearn.datasets) if name.startswith("load")]
	assert "load_{0}".format(input_name) in names, 'Invalid dataset name: ' + input_name + '\nPossible names: \nboston \nwine \niris \ndiabetes \nbreast_cancer'
	
	for i in names:
		if input_name == 'boston':
			from sklearn.datasets import load_boston
			dataset = load_boston()
			classification_flag = False			# For future grouping purposes
		elif input_name == 'wine':
			from sklearn.datasets import load_wine
			dataset = load_wine()		
			classification_flag = True	
		elif input_name == 'iris':
			from sklearn.datasets import load_iris
			dataset = load_iris()
			classification_flag = True
		elif input_name == 'diabetes':
			from sklearn.datasets import load_diabetes
			dataset = load_diabetes()
			classification_flag = False		
		elif input_name == 'breast_cancer':
			from sklearn.datasets import load_breast_cancer
			dataset = load_breast_cancer()
			classification_flag = True	
	return(dataset, classification_flag)

	print('Successfully loaded dataset ', input_name)
	return dataset


def parser_assign():
	'''Setting up parser for the file name and header file name '''
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_name")   # name of the file specified in Dockerfile
	#parser.add_argument("-d", "--header_name", default="no_file", help="name of a headers file") #Optional header file name
	args = parser.parse_args()
	d_name = args.dataset_name
	#if args.header_name:
	#	h_name = args.header_name
	return d_name


def read_data():
	'''Copying data from dataset to Data Frame'''
	data = pd.DataFrame(data = dataset['data'])
	data.columns = dataset['feature_names']			# assigning feature names to the names of the columns
	try:
		data['target'] = pd.Categorical(pd.Series(dataset.target).map(lambda x: dataset.target_names[x]))
	except:
		data['target'] = dataset['target']
	print(data)

	if dataset_name == 'breast_cancer':				# if this is breast cancer dataset
		data1 = data.iloc[:,:10]
		data1['target'] = data['target']
	else: data1 = data
	return data1


def find_mean_std(P):
	'''Calculating mean and std for each of 30 features'''
	ave_feature = np.mean(P) 		
	std_feature = np.std(P) 

	print('\n ave of each measurment:\n', ave_feature)
	print('\n std of each measurment:\n', std_feature)


def plot_3d_clustering (df, columns, f1, f2, f3, folder):
	mal = df.loc[df['target']=='malignant']
	ben = df.loc[df['target']=='benign']

	xm = mal.loc[:,f1]
	ym = mal.loc[:,f2]
	zm = mal.loc[:,f3]


	xb = ben.loc[:,f1]
	yb = ben.loc[:,f2]
	zb = ben.loc[:,f3]

	scatter = dict(
		mode = "markers",
		name = "y",
		type = "scatter3d",    
		x = df[f1], y = df[f2], z = df[f3],
		marker = dict( size=2, color="rgb(23, 190, 207)" )
	)
	clustersm = dict(
		alphahull = 7,
		name = "Malignant",
		opacity = 0.1,
		type = "mesh3d",    
		x = xm, y = ym, z = zm,
		color = 'rgb(0, 128, 128)'
	)
	clustersb = dict(
		alphahull = 7,
		name = "Benign",
		opacity = 0.1,
		type = "mesh3d",    
		x = xb, y = yb, z = zb,
		color = 'rgb(0, 0, 128)'
	)
	layout = dict(
		title = '3d point clustering',
		scene = dict(
			xaxis = dict( zeroline=False, title=f1 ),
			yaxis = dict( zeroline=False, title=f2 ),
			zaxis = dict( zeroline=False, title=f3 ),
		)
	)
	fig = dict( data=[scatter, clustersm, clustersb], layout=layout )
	plot(fig, filename="./{0}/3D_Clustering_{1}_{2}_{3}.html".format(folder,f1,f2,f3), auto_open=True)


def plot_box(df, columns, folder):
	'''Box plot for each feature'''
	mal = df.loc[df['target']=='malignant']
	ben = df.loc[df['target']=='benign']
	l = len(columns)
	n_cols = math.ceil(math.sqrt(l))		#Calculating scaling for any number of features
	n_rows = math.ceil(l / n_cols)
	
	i=0
	for r in range(n_rows):
		for c in range(n_cols):
			if i < len(columns):
				if columns[i] == 'target':
					i=i+1
				else:
					trace0 = go.Box(
						y=ben[columns[i]],
						name = 'Benign',
						boxpoints = 'suspectedoutliers',
						marker = dict(
							color = 'rgb(0, 128, 128)',
							outliercolor = 'rgba(219, 64, 82, 0.6)'
						)
					)
					trace1 = go.Box(
						y=mal[columns[i]],
						name = 'Malignant',
						boxpoints = 'suspectedoutliers',
						marker = dict(
							color = 'rgb(214, 12, 140)',
							outliercolor = 'rgba(219, 64, 82, 0.6)',
						)
					)	
					data = [trace0, trace1]
					layout = go.Layout(
    					yaxis=dict(
        				title=columns[i],
        				zeroline=False
    					),
						showlegend = True,
						height = 700,
						width = 1300,
						title='Box plot grouped by Class(target)'
    					#boxmode='group'
					)
					fig = go.Figure(data=data, layout=layout)
					plot(fig, filename="./{0}/box_plot_{1}.html".format(folder,columns[i]), auto_open=False)
					i=i+1
			


def plot_histograms(df, columns, folder, name):
	'''Histogram all in one figure'''

	l = len(columns)
	n_cols = math.ceil(math.sqrt(l))		#Calculating scaling for any number of features
	n_rows = math.ceil(l / n_cols)
	
	fig=plt.figure(figsize=(11, 6), dpi=100)
	for i, col_name in enumerate(columns):
		if (classification_flag == False):
			ax=fig.add_subplot(n_rows,n_cols,i+1)
			df[col_name].hist(bins=10,ax=ax)
			ax.set_title(col_name)
		elif col_name != 'target':
			ax=fig.add_subplot(n_rows,n_cols,i+1)
			df[col_name].hist(bins=10,ax=ax)
			ax.set_title(col_name)
	fig.tight_layout() 
	plt.savefig("./{0}/all_hist_{1}.png".format(folder,name), bbox_inches='tight')
	plt.show()


def plot_histograms_grouped(df, columns, folder, file_name):
	'''Histogram: all features in one figure grouped by one element'''

	l = len(df.columns)-1
	n_cols = math.ceil(math.sqrt(l))		# Calculating scaling for any number of features
	n_rows = math.ceil(l / n_cols)
	
	fig=plt.figure(figsize=(11, 6), dpi=100)
	
	idx = 0
	for i, col_name in enumerate(df.columns):		# Going through all the features
		idx = idx+1
		if col_name != 'target':				# Avoiding a histogram of the grouping element
			ax=fig.add_subplot(n_rows,n_cols,idx)
			ax.set_title(col_name)
			grouped = df.pivot(columns='target', values=col_name)
			for j, gr_feature_name in enumerate(grouped.columns):			# Going through the values of grouping feature (here malignant and benign)
				grouped[gr_feature_name].hist(alpha=0.5, label=gr_feature_name)
			plt.legend(loc='upper right')
		else: idx = idx-1
	fig.tight_layout() 
	plt.savefig("./{0}/all_hist_grouped_{1}.png".format(folder,file_name), bbox_inches='tight')
	plt.show()


def plot_scatter_3d(df, columns, f1, f2, f3, folder):
	"3D scatter "
	mal = df.loc[df['target']=='malignant']
	ben = df.loc[df['target']=='benign']
	
	fig=plt.figure(figsize=(11, 6), dpi=100)

	ax = fig.add_subplot(111, projection='3d')

	xm = mal.loc[:,f1]
	ym = mal.loc[:,f2]
	zm = mal.loc[:,f3]
	ax.scatter(xm, ym, zm, c='r', marker='^', label='malignant')

	xb = ben.loc[:,f1]
	yb = ben.loc[:,f2]
	zb = ben.loc[:,f3]
	ax.scatter(xb, yb, zb, c='b', marker='o', label='benign')

	ax.set_xlabel(f1)
	ax.set_ylabel(f2)
	ax.set_zlabel(f3)
	ax.legend(loc='upper right')
	plt.savefig(("./{0}/3D_{1}-{2}-{3}.png".format(folder, f1, f2, f3)))
	plt.show()
	plt.close('all')


def plot_scatter(df, f1, f2, folder):
	'''Scatter for each pair of features'''
	mal = df.loc[df['target']=='malignant']
	ben = df.loc[df['target']=='benign']

	xm = mal.loc[:,f1]
	ym = mal.loc[:,f2]
	xb = ben.loc[:,f1]
	yb = ben.loc[:,f2]

	mean_f1 = np.mean(df[f1])
	mean_f2 = np.mean(df[f2])

	fig = plt.figure()
	plt.xlabel(f1)
	plt.ylabel(f2)

	plt.scatter(xm, ym,  color='r', label='Malignant')
	plt.scatter(xb, yb, color='b', label='Benign')
	plt.scatter(mean_f1, mean_f2, color='g', marker='D', label='mean value')
	plt.legend(loc='upper right')
	
	plt.savefig(("./{0}/{1}-{2}.png".format(folder, f1, f2)), bbox_inches='tight')
	plt.close('all')
	

def plot_corr(df, folder, file_n):
	''' Plotting correlations'''
	if classification_flag == True:
		#del df['target']
		df.drop(['target'],axis=1)
		number = len(df.columns)-1
	else: number = len(df.columns)
	cor = df.corr()
	fig = plt.figure(figsize=(11, 11))
	plt.imshow(cor, interpolation='nearest')
	#help(plt.imshow)

	im_ticks = range(number)
	plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
	mask = np.zeros_like(cor)
	mask[np.triu_indices_from(mask)] = True

	plt.xticks(im_ticks, df.columns,  rotation=45)
	#help(plt.xticks)
	plt.yticks(im_ticks, df.columns)
	for i in range(number):
		for j in range(number):
			text = plt.text(j, i, (cor.iloc[i, j]).round(2), ha="center", va="center", color="w")
	plt.colorbar()

	plt.savefig(("./{0}/{1}.png".format(folder,file_n)), bbox_inches='tight')
	plt.close('all')

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

# Assigning dataset name to a local variable
dataset_name = parser_assign()

#Loadind dataset from sklearn
dataset, classification_flag = load_data(dataset_name) 
print('flag: ', classification_flag)

# transrferring sklearn dataset to Data Frame
data = read_data()
print(data.columns)

'''# Calculating summary statistics
find_mean_std(data)


# Plotting histograms
if not os.path.exists('hist'):
	os.makedirs('hist')

print('\n Plotting all histograms into one figure')						#Plotting one histogram for all the features
plot_histograms(data, data.columns, 'hist', dataset_name)
if classification_flag == True:
	print('\n Plotting all histograms into one figure grouped by target')#Plotting one histogram for all the features grouped by diagnosis
	plot_histograms_grouped(data, data.columns, 'hist', dataset_name)


#Plotting Box plot
if dataset_name == 'breast_cancer':
	print('\n Plotting box plots')
	if not os.path.exists('box'):
		os.makedirs('box')
	plot_box(data, data.columns, 'box')


# Plotting correlations heatmap
print('\n Plotting correlation hitmap into /corr/ ')
if not os.path.exists('corr'):
	os.makedirs('corr')
plot_corr(data, 'corr', dataset_name)	# Calculating correlation of 10 features and send them to plot


# Plotting scatter
if dataset_name == 'breast_cancer':
	if not os.path.exists('scatter'):
		os.makedirs('scatter')
	for i in range(len(data.iloc[0])-1):
		j = 1
		for j in range((i+j),len(data.iloc[0])-1):
			col_name1 = data.iloc[:,i].name
			col_name2 = data.iloc[:,j].name
			print('\n Plotting scatter of ', col_name1, 'and ', col_name2, ' into /scatter/')
			plot_scatter(data, col_name1, col_name2, 'scatter')

	
#Plotting 3D scatter and clustering for custom features
if dataset_name == 'breast_cancer':
	if not os.path.exists('3D'):
		os.makedirs('3D')
	print('\n Plotting 3D scatters')
	plot_scatter_3d(data, data.columns, 'mean concave points', 'mean symmetry', 'mean compactness', '3D')
	plot_scatter_3d(data, data.columns, 'mean concave points', 'mean smoothness', 'mean compactness', '3D')
	plot_scatter_3d(data, data.columns, 'mean concave points', 'mean perimeter', 'mean compactness', '3D')
	print('\n Plotting 3D scatters with clustering')
	plot_3d_clustering (data, data.columns, 'mean concave points', 'mean symmetry', 'mean compactness', '3D')
	plot_3d_clustering (data, data.columns, 'mean concave points', 'mean smoothness', 'mean compactness', '3D')
	plot_3d_clustering (data, data.columns, 'mean concave points', 'mean perimeter', 'mean compactness', '3D')'''


print(data.pivot(index='target'))

# Performing KNeighborsClassifier 
if classification_flag == True:
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	proj = pca.fit_transform(dataset.data)
	plt.scatter(proj[:, 0], proj[:, 1], c=dataset.target) 
	plt.colorbar() 
	plt.show()


	print('Performing KNeighborsClassifier ')
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	print('X dataset: ', X.shape, 'y targets: ', y.shape, 'train data shape: ', X_train.shape, 'test data shape: ', X_test.shape)
	
	for n in range(1,11):
		clf = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print('KNeighborsClassifier with {0} neighbors score: '.format(n), metrics.f1_score(y_test,y_pred,average="macro"))

	print(cross_val_score(clf, X, y, cv=5))
	#print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))


# Performing GaussianNB 
if classification_flag == True:
	print('Performing GaussianNB ')
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	print('X dataset: ', X.shape, 'y targets: ', y.shape, 'train data shape: ', X_train.shape, 'test data shape: ', X_test.shape)

	clf = GaussianNB()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print('GaussianNB score: ', metrics.f1_score(y_test,y_pred,average="macro"))

	#print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))



# Performing KNeighborsClassifier for the three chosen columns
'''if dataset_name == 'breast_cancer':
	X = np.empty(shape=[len(dataset.data), 3])
	y = np.empty(shape=[len(dataset.data),])
	k = 0
	for j, c in enumerate(dataset.feature_names):
		if dataset.feature_names[j] == 'mean concave points' or dataset.feature_names[j] == 'mean perimeter' or dataset.feature_names[j] == 'mean compactness': 
			for i, s in enumerate(dataset.data):
				#for j, c in enumerate(dataset.feature_names):
				X[i,k] = dataset.data[i,j]
				y[i] = dataset.target[i]
			k = k+1

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	#print('X dataset: ', X.shape, 'y targets: ', y.shape, 'train data shape: ', X_train.shape, 'test data shape: ', X_test.shape)
	for n in range(1,11):
		clf = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print('KNeighborsClassifier (3 features) with {0} neighbors score: '.format(n), metrics.f1_score(y_test,y_pred,average="macro"))'''





