import pandas as pd
from geopy.geocoders import ArcGIS
import csv
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from shapely.geometry import Point, Polygon
import mapclassify  
import numpy as np
import scipy as sp 
from scipy import stats 
from sklearn.cluster import DBSCAN

BBox = [20.7693,20.9526,39.6013, 39.7278]

def dataPrep(siteName,lastYear):
    df = pd.read_csv(siteName+'.csv')
    df = df.drop_duplicates()
    print('Before: ' + str(df.shape))
    try:
        df = df[(df.latitude < BBox[3]) & (df.latitude > BBox[2]) & (df.longitude < BBox[1]) & (df.longitude > BBox[0])]
    except:
        pass
    df.date = pd.to_datetime(df.date, errors='coerce').dropna()
    if lastYear :
        date_before = pd.to_datetime('2020-12-31')
        date_after = pd.to_datetime('2020-1-1')
        df = df[(df.date < date_before) & (df.date > date_after)]
    else:
        date_after = pd.to_datetime('2015-1-1')
        date_before = pd.to_datetime('2020-12-31')
        df = df[(df.date < date_before) & (df.date > date_after)]
    print('After: ' + str(df.shape))
    return df

def cityVisual(siteName, lastYear):
    sns.set()
    df = dataPrep(siteName, lastYear)
    ioa_im = plt.imread('map.png')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.despine(fig, left=True, bottom=True)
    ax.scatter(df.longitude, df.latitude, zorder=1, alpha=0.9, c='r', s=10)
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    if lastYear:
        ax.set_title('Plotting '+ siteName +"'s spatial data on Ioannina Map in one year period")
        plt.xlabel('From: 2019-Oct-1      To: 2020-Oct-1')
    else:
        ax.set_title('Plotting '+ siteName +"'s spatial data on Ioannina Map")
        plt.xlabel('From: ' + str(df.date.min()) + '     To: ' + str(df.date.max()))
    plt.ylabel('Total data: '+ str(df.shape[0]))
    ax.imshow(ioa_im, zorder=0, extent = BBox)
    sns.scatterplot(x="longitude", y="latitude", linewidth=0,data= df, ax=ax)
    plt.show()
#cityVisual('Efood', False)

def viewPlot(siteName, lastYear):
    df = dataPrep(siteName, lastYear)
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
    df.index = df.date
    GB = df.groupby(by = [df.index.month]).store.count()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.barplot(x = x, y = GB.values ,ax = ax ,palette = 'Set3')
    if lastYear:
        plt.xlabel('From: 2020-Jan-1     To: 2020-Dec-31')
        plt.title(siteName + "'s reviews in one year period")
    else:
        plt.xlabel('From: ' + str(df.date.min()) + '     To: ' + str(df.date.max()))
        plt.title(siteName + "'s reviews ALL data")
    plt.ylabel("Total number of reviews: " + str(df.shape[0]))
    plt.show()
#viewPlot('Efood', True)

def viewTwitterPlot(siteName, lastYear):
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
    df = pd.read_csv(siteName+'.csv')
    #df.date = pd.to_datetime(df.date, errors='coerce').dropna()
    df.date = df.date.astype('datetime64[ns]')
    df.index = df.date
    if lastYear :
        date_before = pd.to_datetime('2020-12-31')
        date_after = pd.to_datetime('2020-1-1')
        df = df[(df.date < date_before) & (df.date > date_after)]
    else:
        date_after = pd.to_datetime('2015-1-1')
        date_before = pd.to_datetime('2019-12-31')
        df = df[(df.date < date_before) & (df.date > date_after)]
    GB = df.groupby(by = [df.index.month]).username.count()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(9,7))
    g = sns.barplot(x = x, y = GB.values ,ax = ax ,palette = 'Set3')
    plt.xlabel('From: 2015-1-1 To: 2019-Dec-31')
    plt.ylabel("Total number of Tweets: " + str(df.shape[0]))
    plt.title(siteName + "'s posts in one year period")
    plt.show()
#viewTwitterPlot('twitter',False)

def GeoJson(siteName, lastYear):
    df = dataPrep(siteName, lastYear)
    #GEO DATA FRAME CREATION
    geometry = [Point(xy) for xy in zip(df.longitude , df.latitude)]
    geo_df = gpd.GeoDataFrame(df ,geometry = geometry)
    geo_df.set_crs(epsg=4326, inplace=True)
    #REGIONS LOAD
    regions = gpd.read_file('geo.geojson')
    print(regions)
    #JOIN GEO_DATA WITH REGIONS
    merged_df = gpd.sjoin(geo_df, regions, how = 'inner', op = 'within')
    temp = merged_df.groupby('index_right').store.count()
    regions['oindex'] = [0,1,2,3,4,5,6,7,8]
    for_plot = regions.merge(temp,how = 'left',left_on='oindex', right_on = 'index_right').fillna(0)
    #PLOT
    f, ax = plt.subplots(1, figsize = (11,9))
    #IMAGE
    ioa_im = plt.imread('map.png')
    ax.imshow(ioa_im, zorder=0, extent = BBox)
    #BINS
    bins = mapclassify.Quantiles(for_plot['store'], k=8).bins
    for i in range(len(bins)):
        bins[i] = bins[i]
    #PLOT SETTINGS AND LABELS
    for_plot.plot(column = 'store', ax =ax, legend = True, alpha = 0.7, edgecolor = 'black',scheme="User_Defined", classification_kwds=dict(bins=bins), cmap = 'RdYlGn')
    #for_plot.plot(column = 'store', ax =ax, legend = True, alpha = 0.7, edgecolor = 'black', legend_kwds={'label': "Total number of reviews by shape"})
    if lastYear:
        plt.xlabel('From: 2020-Jan-1 To: 2020-Dec-31')
        plt.title(siteName + "'s reviews in one year period inside shapeboxes")
    else:
        plt.xlabel('From: ' + str(geo_df.date.min()) + ' to: ' + str(geo_df.date.max()))
        plt.title(siteName + "'s reviews all data inside shapeboxes")
    plt.ylabel("Total number of reviews: " + str(int(for_plot.store.sum())))
    plt.show()
   
#GeoJson('TripAdvisor', False)

def storeShape(siteName, col):
    #only for trip
    df = dataPrep(siteName, False)
    df = df.drop_duplicates(subset = ['store','address'])
    df['avgRating'] = df['avgRating'].apply(lambda x: float(x.replace(",",".")))

    #DATA READING efood
    #df = pd.read_csv(siteName+'.csv')
    #print(df.avgRating)
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    print(df)
    #GEO DATA FRAME CREATION
    geometry = [Point(xy) for xy in zip(df.longitude , df.latitude)]
    geo_df = gpd.GeoDataFrame(df ,geometry = geometry)
    geo_df.set_crs(epsg=4326, inplace=True)
    #REGIONS LOAD
    regions = gpd.read_file('geo.geojson')
    #JOIN GEO_DATA WITH REGIONS
    print(geo_df)
    print(regions)
    merged_df = gpd.sjoin(geo_df, regions, how = 'inner', op = 'within')
    print(merged_df)
    temp = merged_df.groupby('index_right')[col].mean()
    print(temp)
    regions['oindex'] = [0,1,2,3,4,5,6,7,8]
    print(regions)
    for_plot = regions.merge(temp,how = 'left',left_on='oindex', right_on = 'index_right').dropna()
    print(for_plot)
    #PLOT
    f, ax = plt.subplots(1, figsize = (11,9))
    #IMAGE
    BBox = (20.7693,20.9526,39.6013, 39.7278)
    ioa_im = plt.imread('map.png')
    ax.imshow(ioa_im, zorder=0, extent = BBox)
    #BINS
    bins = mapclassify.Quantiles(for_plot[col], k=4).bins
    #PLOT SETTINGS AND LABELS
    for_plot.plot(column = col, ax =ax, legend = True, alpha = 0.7, edgecolor = 'black',scheme="User_Defined", classification_kwds=dict(bins=bins), cmap = 'RdYlGn_r')
    plt.title(siteName + "'s "+ col)
    plt.show()
   
#storeShape('TripAdvisor', 'avgPrice')


def viewPlot2(siteName, lastYear):
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
    df = dataPrep(siteName, lastYear)
    df.index = df.date
    GB = df.groupby(by = [df.index.month]).userRating.mean()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x = x, y = GB.values ,ax = ax ,palette = 'Set3')
    ax.set_ylim(GB.min() - 0.05, GB.max() + 0.05)
    plt.title(siteName + "'s Average User Rating")
    if lastYear:
        plt.xlabel('From: 2020-Jan-1 To: 2020-Dec-31')
    else:
        plt.xlabel('From: ' + str(df.date.min()) + '    To: ' + str(df.date.max()))
    plt.ylabel("Total number of reviews: " + str(df.shape[0]))
    plt.show()
#viewPlot2('TripAdvisor', False)

#ONLY E-FOOD
def favoriteFood(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    gb = df.groupby('type')['NoReviews'].sum()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.barplot(x = gb.index, y = gb.values, ax = ax, palette = 'Set3')
    plt.xticks(rotation=60)
    plt.title("Most popular type of food based on the number of reviews")
    plt.ylabel('E-food')
    plt.show()
#favoriteFood('EfoodStores')

def favoriteFoodPie(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    gb = df.groupby('type')['NoReviews'].sum()
    sns.set(style='whitegrid')
    colors = ['yellow', 'pink', 'green', 'cyan', 'red', 'orange']
    plt.pie(gb.values, labels = gb.index, colors = colors, autopct='%1.0f%%')
    plt.title("Most popular type of food based on the number of reviews")
    plt.show()
#favoriteFoodPie('EfoodStores')
def favoriteTripPie(siteName):
    df = dataPrep(siteName, False)
    df = df.drop_duplicates(subset = ['store','address'])
    df = df.dropna(subset = ['type'])
    df['type'] = df['type'].apply(lambda x: x.split(' ')[3])
    df['type'] = df['type'].replace(['μέρη'],'καφετέριες')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    gb = df.groupby('type')['NoReviews'].sum()
    sns.set(style='whitegrid')
    colors = [ 'green', 'cyan', 'red']
    plt.pie(gb.values, labels = gb.index, colors = colors, autopct='%1.0f%%')
    plt.title("Most popular usage of TripAdvisor")
    plt.show()
#favoriteTripPie('TripAdvisor')

def AvgPriceFood(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    gb = df.groupby('type')['avgPrice'].mean()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.barplot(x = gb.index, y = gb.values, ax = ax, palette = 'Set3')
    plt.xticks(rotation=60)
   #ax.set_ylim(4, 4.8)
    plt.title("Average price grouped by type of food.")
    plt.ylabel('E-food')
    plt.show()
#AvgPriceFood('TripAdvisor')

def AvgQualityFood(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    df = df.groupby('type')['avgFoodQuality'].mean()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.barplot(x = df.index, y = df.values, ax = ax, palette = 'Set3')
    ax.set_ylim(4, 4.8)
    plt.xticks(rotation=60)
    plt.title("Average food quality grouped by type of food.")
    plt.ylabel('E-food')
    plt.show()
#AvgQualityFood('EfoodStores')

def AvgQualitySpeed(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    df = df[df.groupby('type')['store'].transform('count') > 3]
    df = df.groupby('type')['avgSpeedQuality'].mean()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(9,7))
    g = sns.barplot(x = df.index, y = df.values, ax = ax, palette = 'Set3')
    ax.set_ylim(4, 4.8)
    plt.xticks(rotation=60)
    plt.title("Average speed grouped by type of food.")
    plt.ylabel('E-food')
    plt.show()
#AvgQualitySpeed('EfoodStores')

def efoodCorrelations(siteName):
    df = pd.read_csv(siteName+'.csv')
    df = df[df.avgRating > 1]
    df = df[df.NoReviews > 1]
    df = df[['store','type','avgPrice','Noitems','avgRating','avgFoodQuality','avgServiceQuality','avgSpeedQuality']]
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax = 1, annot =True)
    heatmap.set_title('Correlation Heatmap Efood', fontdict={'fontsize':12}, pad=12)
    plt.xticks(rotation=60)
    plt.show()
#efoodCorrelations('EfoodStores')
#efoodCorrelations('EfoodStores', 'avgPrice', 'Noitems')
#efoodCorrelations('EfoodStores', 'avgPrice', 'NoReviews')
#efoodCorrelations('EfoodStores', 'avgRating', 'NoReviews')
#efoodCorrelations('EfoodStores', 'avgPrice', 'avgFoodQuality')


#kanena correlation
def TripCorrelations(siteName, col1, col2):
    df = dataPrep(siteName, False)
    df = df.drop_duplicates(subset = ['store','address'])
    df['avgRating'] = df['avgRating'].apply(lambda x: float(x.replace(",",".")))
    df['NoReviews'] = df['NoReviews'].apply(lambda x: int(x))
    df['avgPrice'] = df['avgPrice'].apply(lambda x: float(x))
    df = df.drop_duplicates(subset = ['store','address'])
    df = df[df.avgRating > 1]
    df = df[df.NoReviews > 1]
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax = 1, annot =True)
    heatmap.set_title('Correlation Heatmap Efood', fontdict={'fontsize':12}, pad=12)
    plt.xticks(rotation=60)
    plt.show()
#efoodCorrelations('EfoodStores')

def dbscan(siteName, lastYear, aktina, minSampl):
    df = dataPrep(siteName, lastYear)
    df = df.drop_duplicates(subset = ['store', 'address'])
    coords = df[['latitude', 'longitude']].values.tolist()
    kms_per_radian = 6371.0088
    epsilon = aktina / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples = minSampl, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    #clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    f, ax = plt.subplots(1, figsize = (10,8))
    #IMAGE
    ioa_im = plt.imread('map.png')
    ax.imshow(ioa_im, zorder=0, extent = BBox)
    g = sns.scatterplot(x = df['longitude'], y = df['latitude'], hue = cluster_labels, palette = 'Set1', ax=ax)
    plt.title('DBscan of '+ siteName + '\n ball_tree algorithm and haversine metric used.')
    plt.xlabel('Aktina: ' + str(aktina) + '       Min. Samples: '+ str(minSampl))
    plt.show()
#dbscan('TripAdvisor', False, 0.5, 2)

def differentUsers(siteName):
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
    df = dataPrep(siteName, True)
    df.index = df.date
    GB = df.groupby(by = [df.index.month]).username.nunique()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x = x, y = GB.values ,ax = ax ,palette = 'Set3')
    ax.set_ylim(GB.min() - 6, GB.max() + 6)
    for i, v in enumerate(GB.values):
        plt.text(i -0.18, v + 0.5, str(v))
    plt.title('Different users every month')
    plt.ylabel('TripAdvisor')
    plt.xlabel('From: 2020-Jan-1 To: 2020-Dec-31')
    plt.show()
#differentUsers('Efood')

def differentUsersTwitter(siteName,lastYear):
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
    df = pd.read_csv(siteName+'.csv')
    df.date = df.date.astype('datetime64[ns]')
    
    if lastYear :
        date_before = pd.to_datetime('2020-12-31')
        date_after = pd.to_datetime('2020-1-1')
        df = df[(df.date < date_before) & (df.date > date_after)]
    else:
        date_after = pd.to_datetime('2015-1-1')
        date_before = pd.to_datetime('2020-12-31')
        df = df[(df.date < date_before) & (df.date > date_after)]
    df.index = df.date
    GB = df.groupby(by = [df.index.month]).username.nunique()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x = x, y = GB.values ,ax = ax ,palette = 'Set3')
    ax.set_ylim(GB.min() - 6, GB.max() + 6)
    for i, v in enumerate(GB.values):
        plt.text(i -0.18, v + 0.5, str(v))
    plt.title('Different users every month')
    plt.ylabel('Twitter')
    plt.xlabel('From: 2020-Jan-1 To: 2020-Dec-31')
    plt.show()
#differentUsersTwitter('twitter',True)

def usagePerYear(siteName):
    x = ['2015', '2016', '2017', '2018', '2019']
    df = dataPrep(siteName, False)
    df.index = df.date
    
    GB = df.groupby(by = [df.index.year]).username.count()
    print(GB)
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x = GB.index, y = GB.values ,ax = ax ,palette = 'Set3')
    plt.title('Percentage difference with previous year.')
    plt.ylabel(siteName)
    plt.xlabel('Year')
    for i, v in enumerate(GB.values):
        if i > 0:
            increase = GB.values[i]-GB.values[i-1]
            percentage = increase/GB.values[i-1]*100
            percentage = round(percentage)
            plt.text(i -0.18, v + 0.5, str(percentage)+'%')
    plt.show()
#usagePerYear('TripAdvisor')

def usagePerYear1(siteName):
    x = ['2015', '2016', '2017', '2018', '2019', '2020']
    df = pd.read_csv(siteName+'.csv')
    df.date = df.date.astype('datetime64[ns]')
    date_after = pd.to_datetime('2015-1-1')
    date_before = pd.to_datetime('2019-12-31')
    df = df[(df.date < date_before) & (df.date > date_after)]
    df.index = df.date
    GB = df.groupby(by = [df.index.year]).username.count()
    print(GB)
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x = GB.index, y = GB.values ,ax = ax ,palette = 'Set3')
    plt.title('Percentage difference with previous year.')
    plt.ylabel(siteName)
    plt.xlabel('Year')
    for i, v in enumerate(GB.values):
        if i > 0:
            increase = GB.values[i]-GB.values[i-1]
            percentage = increase/GB.values[i-1]*100
            percentage = round(percentage)
            plt.text(i -0.18, v + 0.5, str(percentage)+'%')
    plt.show()
#usagePerYear1('Twitter')


def ReviewRating(siteName):
    df = dataPrep(siteName, False)
    #df = df[(df.latitude < 39.7006) & (df.latitude > 39.6013) & (df.longitude < 20.9526) & (df.longitude > 20.7659)]
    #df = df[df.groupby('type')['store'].transform('count') > 3]
    df = df.dropna(subset=['content'])
    df.userRating = df.userRating.apply(lambda x: round(x))
    gb = df.groupby('userRating')['store'].count()
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.barplot(x = gb.index, y = gb.values, ax = ax, palette = 'Set3')
    #plt.xticks(rotation=60)
    plt.title("Distribution of rounded ratings")
    plt.ylabel('E-food')
    plt.xlabel("User's ratings")
    plt.show()

#ReviewRating('Efood')