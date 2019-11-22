
<h1 align=center><font size = 6>Segmenting and Clustering Neighborhoods in Toronto</font></h1>

## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>

1. <a href="#item1">First part of assignment</a>

2. <a href="#item2">Second part of assignment</a>

3. <a href="#item3">Third part of assignment</a>

</font>

Let us download all the dependencies and libraries that we need for this assignment


```python
import numpy as np
import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

!conda install -c conda-forge lxml --yes  #helps in scraping the web for data


import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

!conda install -c conda-forge folium=0.5.0 --yes 
import folium # map rendering library

print('Libraries imported.')
```

    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/conda/envs/Python36
    
      added / updated specs: 
        - lxml
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        certifi-2019.9.11          |           py36_0         147 KB  conda-forge
        ca-certificates-2019.9.11  |       hecc5488_0         144 KB  conda-forge
        lxml-4.4.1                 |   py36h7ec2d77_0         1.6 MB  conda-forge
        openssl-1.1.1d             |       h516909a_0         2.1 MB  conda-forge
        ------------------------------------------------------------
                                               Total:         3.9 MB
    
    The following packages will be UPDATED:
    
        certifi:         2019.9.11-py36_0     --> 2019.9.11-py36_0     conda-forge
        lxml:            4.3.1-py36hefd8a0e_0 --> 4.4.1-py36h7ec2d77_0 conda-forge
    
    The following packages will be DOWNGRADED:
    
        ca-certificates: 2019.10.16-0         --> 2019.9.11-hecc5488_0 conda-forge
        openssl:         1.1.1d-h7b6447c_3    --> 1.1.1d-h516909a_0    conda-forge
    
    
    Downloading and Extracting Packages
    certifi-2019.9.11    | 147 KB    | ##################################### | 100% 
    ca-certificates-2019 | 144 KB    | ##################################### | 100% 
    lxml-4.4.1           | 1.6 MB    | ##################################### | 100% 
    openssl-1.1.1d       | 2.1 MB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/conda/envs/Python36
    
      added / updated specs: 
        - folium=0.5.0
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        altair-3.2.0               |           py36_0         770 KB  conda-forge
        branca-0.3.1               |             py_0          25 KB  conda-forge
        vincent-0.4.4              |             py_1          28 KB  conda-forge
        folium-0.5.0               |             py_0          45 KB  conda-forge
        ------------------------------------------------------------
                                               Total:         868 KB
    
    The following NEW packages will be INSTALLED:
    
        altair:  3.2.0-py36_0 conda-forge
        branca:  0.3.1-py_0   conda-forge
        folium:  0.5.0-py_0   conda-forge
        vincent: 0.4.4-py_1   conda-forge
    
    
    Downloading and Extracting Packages
    altair-3.2.0         | 770 KB    | ##################################### | 100% 
    branca-0.3.1         | 25 KB     | ##################################### | 100% 
    vincent-0.4.4        | 28 KB     | ##################################### | 100% 
    folium-0.5.0         | 45 KB     | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Libraries imported.


<a id='item1'></a>

## 1. First part of the assignment

#### Extracting data from a website
We are going to download and clean the Toronto neighborhoods in cells below


```python
df=pd.read_html("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
df=df[0]  # Taking the dataframe with the required data
df=df[df.Borough!='Not assigned']    #ignoring the rows with the non-assigned rows
df.reset_index(inplace=True)
df.drop(columns='index', inplace=True)
df.Neighbourhood[df.Neighbourhood=='Not assigned']=df.Borough[df.Neighbourhood=='Not assigned']  #Naming the  non-assigned neighborhoods
                                                                                                #with the borough names
```


```python
#Grouping neighborhoods with same postal codes and separting them with a comma
df_table=df.groupby(['Postcode','Borough'])['Neighbourhood'].apply(', '.join).reset_index()

df_table.columns=['PostalCode', 'Borough', 'Neighborhood']
df_table.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1S</td>
      <td>Scarborough</td>
      <td>Agincourt</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M1T</td>
      <td>Scarborough</td>
      <td>Clarks Corners, Sullivan, Tam O'Shanter</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1V</td>
      <td>Scarborough</td>
      <td>Agincourt North, L'Amoreaux East, Milliken, St...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M1W</td>
      <td>Scarborough</td>
      <td>L'Amoreaux West</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M1X</td>
      <td>Scarborough</td>
      <td>Upper Rouge</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
    </tr>
  </tbody>
</table>
</div>




```python
shape=df_table.shape
print("The number of rows in the above dataframe is",shape[0])
```

    The number of rows in the above dataframe is 103


<a id='item2'></a>

## 2. Second part of the assignment

#### Finding out the coordinates of the neighborhoods
We will download the coordinates values based on the Postal Codes and merge them with the above table in the cells below


```python
coord=pd.read_csv("https://cocl.us/Geospatial_data")  #Extracting the data from the given website
coord.set_index('Postal Code', inplace=True)
coord.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
    <tr>
      <th>Postal Code</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M1B</th>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>M1C</th>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>M1E</th>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>M1G</th>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>M1H</th>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merging the coordinates data with the borough and neighborhood data

temp=coord.loc[df_table.PostalCode]
temp.reset_index(inplace=True)
temp.head()
df_table[['Latitude','Longitude']]=temp[['Latitude','Longitude']]
df_table.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1S</td>
      <td>Scarborough</td>
      <td>Agincourt</td>
      <td>43.794200</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M1T</td>
      <td>Scarborough</td>
      <td>Clarks Corners, Sullivan, Tam O'Shanter</td>
      <td>43.781638</td>
      <td>-79.304302</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1V</td>
      <td>Scarborough</td>
      <td>Agincourt North, L'Amoreaux East, Milliken, St...</td>
      <td>43.815252</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M1W</td>
      <td>Scarborough</td>
      <td>L'Amoreaux West</td>
      <td>43.799525</td>
      <td>-79.318389</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M1X</td>
      <td>Scarborough</td>
      <td>Upper Rouge</td>
      <td>43.836125</td>
      <td>-79.205636</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
      <td>43.803762</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
      <td>43.778517</td>
      <td>-79.346556</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
      <td>43.786947</td>
      <td>-79.385975</td>
    </tr>
  </tbody>
</table>
</div>



<a id='item3'></a>

## 3. Third part of the assignment

#### We will be exploring and clustering the neighborhoods whose boroughs' have the word "Toronto" in them


```python
#Separating the above dataframe such that the borough's have the word "Toronto" in them

df_toronto=df_table[df_table['Borough'].str.find('Toronto')>0]
df_toronto.reset_index(inplace=True)
df_toronto=df_toronto.drop('index', axis=1)
df_toronto.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
  </tbody>
</table>
</div>



##### We will display the above neighborhoods on a map in the cells below


```python
#Obtaining the coordnitaes for centering the map

latitude=df_toronto['Latitude'].mean()
longitude=df_toronto['Longitude'].mean()
print('The central geographical coordinate for Boroughs which contain the  word toronto are {}, {}.'.format(latitude, longitude))
```

    The central geographical coordinate for Boroughs which contain the  word toronto are 43.66726218421052, -79.38988323421053.



```python
# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_toronto['Latitude'], df_toronto['Longitude'], df_toronto['Borough'], df_toronto['Neighborhood']):
    label = borough+": "+neighborhood   #'{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjY3MjYyMTg0MjEwNTIsLTc5LjM4OTg4MzIzNDIxMDUzXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDEyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl83MGNjNTRiYTg5YmY0MDg2OTQ0Y2U0M2NmMmZmNzM0OCA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWM3NzAxMzYwN2Y1NGNjZmI1MGVjODM5YmVhMDRjMGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NmJlNWU5YmFhMjE0YWU1OWNiZGI5YzE4MWQ2N2I0YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZWQ0MTMwZDBhZWU0YjRkYTZjOGVhYjE4ZDc2MzlmZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGVkNDEzMGQwYWVlNGI0ZGE2YzhlYWIxOGQ3NjM5ZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250bzogVGhlIEJlYWNoZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY2YmU1ZTliYWEyMTRhZTU5Y2JkYjljMTgxZDY3YjRhLnNldENvbnRlbnQoaHRtbF9kZWQ0MTMwZDBhZWU0YjRkYTZjOGVhYjE4ZDc2MzlmZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85Yzc3MDEzNjA3ZjU0Y2NmYjUwZWM4MzliZWEwNGMwYy5iaW5kUG9wdXAocG9wdXBfNjZiZTVlOWJhYTIxNGFlNTljYmRiOWMxODFkNjdiNGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWEwNjZkZGY1MDEwNGJhNWJhZDE2NDRmNDVjMmYzYzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWM4ZTdmZmIxYzZiNGJlOWI0NzM2MzhmNTljMDlkOWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTljNzM3MWU2Y2MwNDY5OWEzOGY4YzNjMjUxMTFhM2YgPSAkKCc8ZGl2IGlkPSJodG1sXzU5YzczNzFlNmNjMDQ2OTlhMzhmOGMzYzI1MTExYTNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG86IFRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjOGU3ZmZiMWM2YjRiZTliNDczNjM4ZjU5YzA5ZDlkLnNldENvbnRlbnQoaHRtbF81OWM3MzcxZTZjYzA0Njk5YTM4ZjhjM2MyNTExMWEzZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYTA2NmRkZjUwMTA0YmE1YmFkMTY0NGY0NWMyZjNjNS5iaW5kUG9wdXAocG9wdXBfYWM4ZTdmZmIxYzZiNGJlOWI0NzM2MzhmNTljMDlkOWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTNkM2YzZjMzY2Q5NGY3N2JkYzQxNmE5NTNkOTc0NzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MjkzZTBhMjFkYjM0OWM5YTIzM2FlNzBhYjJiZmM2YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYjM0MzA5M2ZiNzM0NmUxOWRmNWNmYTRhMWQ4YmIyOCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGIzNDMwOTNmYjczNDZlMTlkZjVjZmE0YTFkOGJiMjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250bzogVGhlIEJlYWNoZXMgV2VzdCwgSW5kaWEgQmF6YWFyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MjkzZTBhMjFkYjM0OWM5YTIzM2FlNzBhYjJiZmM2YS5zZXRDb250ZW50KGh0bWxfZGIzNDMwOTNmYjczNDZlMTlkZjVjZmE0YTFkOGJiMjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTNkM2YzZjMzY2Q5NGY3N2JkYzQxNmE5NTNkOTc0NzYuYmluZFBvcHVwKHBvcHVwXzUyOTNlMGEyMWRiMzQ5YzlhMjMzYWU3MGFiMmJmYzZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg2NmFmN2VjODU3OTQyMTViN2NiZDU5OWQ2ZTUxYjIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNjZWY3NTQyMTFiODRiMzI5MGZiMDhjY2I5YWE0YjkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y4YjY3ZGI4NjhlNDRmNWU4MWQ5NTZjN2U1MzJjYzQ4ID0gJCgnPGRpdiBpZD0iaHRtbF9mOGI2N2RiODY4ZTQ0ZjVlODFkOTU2YzdlNTMyY2M0OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvOiBTdHVkaW8gRGlzdHJpY3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNjZWY3NTQyMTFiODRiMzI5MGZiMDhjY2I5YWE0YjkyLnNldENvbnRlbnQoaHRtbF9mOGI2N2RiODY4ZTQ0ZjVlODFkOTU2YzdlNTMyY2M0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NjZhZjdlYzg1Nzk0MjE1YjdjYmQ1OTlkNmU1MWIyMS5iaW5kUG9wdXAocG9wdXBfM2NlZjc1NDIxMWI4NGIzMjkwZmIwOGNjYjlhYTRiOTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDEzMjEzODc4Y2Y5NDVlMDk1NzMzMjQzMmM1MzIwMzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjgwMjA1LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QxNzViZTExYjI4NzRkYTM5NzkwMjhlZmRkMzNhMDZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA2ZGUzMGE0NDhhNjRiODA5MzBiMjY3NWU4ODU3YjM1ID0gJCgnPGRpdiBpZD0iaHRtbF8wNmRlMzBhNDQ4YTY0YjgwOTMwYjI2NzVlODg1N2IzNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBMYXdyZW5jZSBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMTc1YmUxMWIyODc0ZGEzOTc5MDI4ZWZkZDMzYTA2Yi5zZXRDb250ZW50KGh0bWxfMDZkZTMwYTQ0OGE2NGI4MDkzMGIyNjc1ZTg4NTdiMzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDEzMjEzODc4Y2Y5NDVlMDk1NzMzMjQzMmM1MzIwMzQuYmluZFBvcHVwKHBvcHVwX2QxNzViZTExYjI4NzRkYTM5NzkwMjhlZmRkMzNhMDZiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M1N2VkMDAwNTUzNzQ2ZWJiZTM1NjVjMjNhODIyYWU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZDg4ZmZlYTFhZWY0ZDVhOWNjY2U1ZGMzY2JkNzBmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NjY1YjY2NjY4ODY0MmU0OTdlZWM2YWJmYjQyM2RmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNzY2NWI2NjY2ODg2NDJlNDk3ZWVjNmFiZmI0MjNkZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bzogRGF2aXN2aWxsZSBOb3J0aDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmQ4OGZmZWExYWVmNGQ1YTljY2NlNWRjM2NiZDcwZjEuc2V0Q29udGVudChodG1sXzc2NjViNjY2Njg4NjQyZTQ5N2VlYzZhYmZiNDIzZGZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M1N2VkMDAwNTUzNzQ2ZWJiZTM1NjVjMjNhODIyYWU5LmJpbmRQb3B1cChwb3B1cF82ZDg4ZmZlYTFhZWY0ZDVhOWNjY2U1ZGMzY2JkNzBmMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NTcyNDRiZTY2ZGE0MTU5OWIxMjdkMjAwOTgwMmU2NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNTM4MzQsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJiNWFmMDA5MTI2NzQ4NzhhYzMwYjkyMmYyMzAyOWE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3NmEwNDA5MmMxZTQ5YTk5OWE4MmE2N2UwYTE2MWJmID0gJCgnPGRpdiBpZD0iaHRtbF82NzZhMDQwOTJjMWU0OWE5OTlhODJhNjdlMGExNjFiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBOb3J0aCBUb3JvbnRvIFdlc3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJiNWFmMDA5MTI2NzQ4NzhhYzMwYjkyMmYyMzAyOWE2LnNldENvbnRlbnQoaHRtbF82NzZhMDQwOTJjMWU0OWE5OTlhODJhNjdlMGExNjFiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NTcyNDRiZTY2ZGE0MTU5OWIxMjdkMjAwOTgwMmU2NC5iaW5kUG9wdXAocG9wdXBfMmI1YWYwMDkxMjY3NDg3OGFjMzBiOTIyZjIzMDI5YTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmEyNzViOTQ3MzEzNGViYmI1ZjQ1ZWNhNjM4MTg3NmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FmYzMzMDJhOWQ2MDQzMzNhMTNlZGEwZDM5MzE1NTI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q1Yzk3ZDg4YzBiNjQyNThhYjM3Y2EyMzMxYmU0ZDFmID0gJCgnPGRpdiBpZD0iaHRtbF9kNWM5N2Q4OGMwYjY0MjU4YWIzN2NhMjMzMWJlNGQxZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBEYXZpc3ZpbGxlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZmMzMzAyYTlkNjA0MzMzYTEzZWRhMGQzOTMxNTUyNC5zZXRDb250ZW50KGh0bWxfZDVjOTdkODhjMGI2NDI1OGFiMzdjYTIzMzFiZTRkMWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmEyNzViOTQ3MzEzNGViYmI1ZjQ1ZWNhNjM4MTg3NmMuYmluZFBvcHVwKHBvcHVwX2FmYzMzMDJhOWQ2MDQzMzNhMTNlZGEwZDM5MzE1NTI0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VkNjFkYzRhYzg4ZjRhNzE4MTRiNzRlOTI1ZTZkYmFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjI0MTA5NDU5Y2I1NGIwNWIxMzg5NWUzNWUyYjBkNTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDY3ZTU2MTY3MmM2NDdmNWEzMTg4NDlkMDFlNzU3OGIgPSAkKCc8ZGl2IGlkPSJodG1sX2Q2N2U1NjE2NzJjNjQ3ZjVhMzE4ODQ5ZDAxZTc1NzhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG86IE1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjI0MTA5NDU5Y2I1NGIwNWIxMzg5NWUzNWUyYjBkNTguc2V0Q29udGVudChodG1sX2Q2N2U1NjE2NzJjNjQ3ZjVhMzE4ODQ5ZDAxZTc1NzhiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VkNjFkYzRhYzg4ZjRhNzE4MTRiNzRlOTI1ZTZkYmFlLmJpbmRQb3B1cChwb3B1cF9mMjQxMDk0NTljYjU0YjA1YjEzODk1ZTM1ZTJiMGQ1OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNjQ0YThmMmZlZjg0NmM5OWVjZjVkY2IyNWI3NWVlZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZjN2ZkYTRkMzViZTRiNjU5ZjljMGNiYTY4NDA2NjVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QyZDk0NjcwNWM2YzQ0YWNhZWUxNDc0YzQwMDU2MWM2ID0gJCgnPGRpdiBpZD0iaHRtbF9kMmQ5NDY3MDVjNmM0NGFjYWVlMTQ3NGM0MDA1NjFjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBEZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIFN1bW1lcmhpbGwgV2VzdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmM3ZmRhNGQzNWJlNGI2NTlmOWMwY2JhNjg0MDY2NWUuc2V0Q29udGVudChodG1sX2QyZDk0NjcwNWM2YzQ0YWNhZWUxNDc0YzQwMDU2MWM2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y2NDRhOGYyZmVmODQ2Yzk5ZWNmNWRjYjI1Yjc1ZWVkLmJpbmRQb3B1cChwb3B1cF9mYzdmZGE0ZDM1YmU0YjY1OWY5YzBjYmE2ODQwNjY1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMWU1MDVhZmJkNmY0YjQ0YjYxNTE5NzBhNWU4YzgxZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4ZDE0OTEzMmUwNDQ2NjViZTY1ZTg2YzQ5ZjMxNGNkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgyOGE2OGMyNDVkMTQ4YmQ4MzcxNGNiNjhlMTcyNzllID0gJCgnPGRpdiBpZD0iaHRtbF84MjhhNjhjMjQ1ZDE0OGJkODM3MTRjYjY4ZTE3Mjc5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogUm9zZWRhbGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4ZDE0OTEzMmUwNDQ2NjViZTY1ZTg2YzQ5ZjMxNGNkLnNldENvbnRlbnQoaHRtbF84MjhhNjhjMjQ1ZDE0OGJkODM3MTRjYjY4ZTE3Mjc5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMWU1MDVhZmJkNmY0YjQ0YjYxNTE5NzBhNWU4YzgxZS5iaW5kUG9wdXAocG9wdXBfZjhkMTQ5MTMyZTA0NDY2NWJlNjVlODZjNDlmMzE0Y2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGIyMWQ5MGM3ZDU1NDQwM2JhMGQ3N2ZmNDk5ZTVkYTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWZmN2RlMTAzMjIyNDZlYzhlOGFhMGI5N2FlZjlkODEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjg2N2MzMWE1ZmNmNDc5MzhkMmY1NWUwZGJiZjcyOTIgPSAkKCc8ZGl2IGlkPSJodG1sX2I4NjdjMzFhNWZjZjQ3OTM4ZDJmNTVlMGRiYmY3MjkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBDYWJiYWdldG93biwgU3QuIEphbWVzIFRvd248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VmZjdkZTEwMzIyMjQ2ZWM4ZThhYTBiOTdhZWY5ZDgxLnNldENvbnRlbnQoaHRtbF9iODY3YzMxYTVmY2Y0NzkzOGQyZjU1ZTBkYmJmNzI5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjIxZDkwYzdkNTU0NDAzYmEwZDc3ZmY0OTllNWRhNi5iaW5kUG9wdXAocG9wdXBfZWZmN2RlMTAzMjIyNDZlYzhlOGFhMGI5N2FlZjlkODEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTBhZmM3Mjk5YjQ2NDJmOWI1NWExZTVkOTdkZDJjMDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZjYyMjM0YzU4MzE0MmZkODYwOGE4ZWI3NTdlOGY2MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMDBiM2UxOWU3ZDg0ZGJhOWNkMjA0MThlNmYyMDdjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfZDAwYjNlMTllN2Q4NGRiYTljZDIwNDE4ZTZmMjA3Y2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENodXJjaCBhbmQgV2VsbGVzbGV5PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZjYyMjM0YzU4MzE0MmZkODYwOGE4ZWI3NTdlOGY2MC5zZXRDb250ZW50KGh0bWxfZDAwYjNlMTllN2Q4NGRiYTljZDIwNDE4ZTZmMjA3Y2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTBhZmM3Mjk5YjQ2NDJmOWI1NWExZTVkOTdkZDJjMDguYmluZFBvcHVwKHBvcHVwXzdmNjIyMzRjNTgzMTQyZmQ4NjA4YThlYjc1N2U4ZjYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUxZTM3NjI3Mjk1ZDQ1MTNhNWEwOWE2Yjk5YTA0MjU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85N2UzMTIzOWVlNzk0ZWYxYjBmNTJjYzg5MzA4NzkzZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YzI5NWFkMzIyMGQ0YjRiOTQyNTA1MzNhOGZjNzc5NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNWMyOTVhZDMyMjBkNGI0Yjk0MjUwNTMzYThmYzc3OTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IEhhcmJvdXJmcm9udDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTdlMzEyMzllZTc5NGVmMWIwZjUyY2M4OTMwODc5M2Quc2V0Q29udGVudChodG1sXzVjMjk1YWQzMjIwZDRiNGI5NDI1MDUzM2E4ZmM3Nzk0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUxZTM3NjI3Mjk1ZDQ1MTNhNWEwOWE2Yjk5YTA0MjU4LmJpbmRQb3B1cChwb3B1cF85N2UzMTIzOWVlNzk0ZWYxYjBmNTJjYzg5MzA4NzkzZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NTYwMGI0Njk4NWI0NTI1OGUwZDJkN2U0ZWJmYjY2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgzYzRkNjRlM2M5YTQ2ZTNiODQ4MzEzNjdlODA3YTZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllOTBiMGQ1OWM2MDQ0ODU5MDg1MGZmY2UxMDlhMTYwID0gJCgnPGRpdiBpZD0iaHRtbF85ZTkwYjBkNTljNjA0NDg1OTA4NTBmZmNlMTA5YTE2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogUnllcnNvbiwgR2FyZGVuIERpc3RyaWN0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84M2M0ZDY0ZTNjOWE0NmUzYjg0ODMxMzY3ZTgwN2E2Zi5zZXRDb250ZW50KGh0bWxfOWU5MGIwZDU5YzYwNDQ4NTkwODUwZmZjZTEwOWExNjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzU2MDBiNDY5ODViNDUyNThlMGQyZDdlNGViZmI2NjEuYmluZFBvcHVwKHBvcHVwXzgzYzRkNjRlM2M5YTQ2ZTNiODQ4MzEzNjdlODA3YTZmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgyZWE0YTMyYjcxNTQyNGI5ZWZkOGQwZjZlNWUwOTRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYzFmNjk3ZGJjOTk0Mjc1YmI0N2QwNzcxZjY3ZGYzMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jM2EzNWZmZDg1MDE0Y2MwYmYwYTc0MjgwZDU2ZWUxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzNhMzVmZmQ4NTAxNGNjMGJmMGE3NDI4MGQ1NmVlMWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IFN0LiBKYW1lcyBUb3duPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYzFmNjk3ZGJjOTk0Mjc1YmI0N2QwNzcxZjY3ZGYzMy5zZXRDb250ZW50KGh0bWxfYzNhMzVmZmQ4NTAxNGNjMGJmMGE3NDI4MGQ1NmVlMWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODJlYTRhMzJiNzE1NDI0YjllZmQ4ZDBmNmU1ZTA5NGEuYmluZFBvcHVwKHBvcHVwXzFjMWY2OTdkYmM5OTQyNzViYjQ3ZDA3NzFmNjdkZjMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IyNjYwZmViYjg1NDQxZjg5MWRmM2VhMjBhYjhlNDhlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IxZDg3ODNhMDhjMjQxNGE4ZGJlNjMwZGM1ZTM1OTIyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzZWExZjZiYmJkMzRhN2NhNWM5Nzc1M2IzNzZmYjliID0gJCgnPGRpdiBpZD0iaHRtbF9hM2VhMWY2YmJiZDM0YTdjYTVjOTc3NTNiMzc2ZmI5YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogQmVyY3p5IFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IxZDg3ODNhMDhjMjQxNGE4ZGJlNjMwZGM1ZTM1OTIyLnNldENvbnRlbnQoaHRtbF9hM2VhMWY2YmJiZDM0YTdjYTVjOTc3NTNiMzc2ZmI5Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMjY2MGZlYmI4NTQ0MWY4OTFkZjNlYTIwYWI4ZTQ4ZS5iaW5kUG9wdXAocG9wdXBfYjFkODc4M2EwOGMyNDE0YThkYmU2MzBkYzVlMzU5MjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmIyNjNhZTRjYTEwNGYxYzhlNmYzNjY4ZmE4YzhjZTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JlNjU3Y2I1ODJkMTQ5YzBiYTU4YmY4NjQxZTVjMGE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNkMGJmOWFjNjk1MjQ0OGI5ODFlZDcwNWI3MDU4YzJjID0gJCgnPGRpdiBpZD0iaHRtbF8zZDBiZjlhYzY5NTI0NDhiOTgxZWQ3MDViNzA1OGMyYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogQ2VudHJhbCBCYXkgU3RyZWV0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZTY1N2NiNTgyZDE0OWMwYmE1OGJmODY0MWU1YzBhNi5zZXRDb250ZW50KGh0bWxfM2QwYmY5YWM2OTUyNDQ4Yjk4MWVkNzA1YjcwNThjMmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmIyNjNhZTRjYTEwNGYxYzhlNmYzNjY4ZmE4YzhjZTkuYmluZFBvcHVwKHBvcHVwX2JlNjU3Y2I1ODJkMTQ5YzBiYTU4YmY4NjQxZTVjMGE2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YyOGUzY2JhMjk5YzRmMWQ5ZjhkOWY0NjM0OWJiM2M3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmIwOWE5OWZkYmMwNGJkN2EyZjk0YjU5ZDFhMTJlZTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGE5ZDQ1M2RlMTMyNGIxODhmM2E4YmQ3ZmYzY2ZiNzMgPSAkKCc8ZGl2IGlkPSJodG1sXzhhOWQ0NTNkZTEzMjRiMTg4ZjNhOGJkN2ZmM2NmYjczIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBBZGVsYWlkZSwgS2luZywgUmljaG1vbmQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMDlhOTlmZGJjMDRiZDdhMmY5NGI1OWQxYTEyZWUyLnNldENvbnRlbnQoaHRtbF84YTlkNDUzZGUxMzI0YjE4OGYzYThiZDdmZjNjZmI3Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMjhlM2NiYTI5OWM0ZjFkOWY4ZDlmNDYzNDliYjNjNy5iaW5kUG9wdXAocG9wdXBfZmIwOWE5OWZkYmMwNGJkN2EyZjk0YjU5ZDFhMTJlZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2E3YjgzNDIyNjRmNGU4MjkwYjBlZjU0OWQ1OWRhODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMTM2MTJkOWE5ZTg0NzdiOTAzNWNhZDNmNDYxOGNhNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NjJmMWU5ZDA0MjA0NmNkYWI3MWYxZDBhMzQ0MThlMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNzYyZjFlOWQwNDIwNDZjZGFiNzFmMWQwYTM0NDE4ZTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IEhhcmJvdXJmcm9udCBFYXN0LCBUb3JvbnRvIElzbGFuZHMsIFVuaW9uIFN0YXRpb248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxMzYxMmQ5YTllODQ3N2I5MDM1Y2FkM2Y0NjE4Y2E0LnNldENvbnRlbnQoaHRtbF83NjJmMWU5ZDA0MjA0NmNkYWI3MWYxZDBhMzQ0MThlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYTdiODM0MjI2NGY0ZTgyOTBiMGVmNTQ5ZDU5ZGE4NC5iaW5kUG9wdXAocG9wdXBfMzEzNjEyZDlhOWU4NDc3YjkwMzVjYWQzZjQ2MThjYTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWY3YzZkMjQ1ZGNmNDNjY2JkMDBhZWMyNjJmOWZhZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iOWIxYzEwYjdhZjU0MzQyYTY3NmEzY2FhNDA0Yjk0MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lY2NhMzUwODYyMDc0MmM4YWQwOGI4YTBjZmQ0N2VkZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWNjYTM1MDg2MjA3NDJjOGFkMDhiOGEwY2ZkNDdlZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IERlc2lnbiBFeGNoYW5nZSwgVG9yb250byBEb21pbmlvbiBDZW50cmU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I5YjFjMTBiN2FmNTQzNDJhNjc2YTNjYWE0MDRiOTQwLnNldENvbnRlbnQoaHRtbF9lY2NhMzUwODYyMDc0MmM4YWQwOGI4YTBjZmQ0N2VkZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZjdjNmQyNDVkY2Y0M2NjYmQwMGFlYzI2MmY5ZmFmYy5iaW5kUG9wdXAocG9wdXBfYjliMWMxMGI3YWY1NDM0MmE2NzZhM2NhYTQwNGI5NDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWNkNjdhMmIyOTZhNDE1OTg4ZjEzYTYwODYzOTRjY2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMTBhMmMyNTFiNjE0MTlhYmNjYzY4ZTgzYWIwM2JhNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjcxMGY1NTBlZmQ0YmNhYjUwZTU4NzA3ODM0YzAzMiA9ICQoJzxkaXYgaWQ9Imh0bWxfODI3MTBmNTUwZWZkNGJjYWI1MGU1ODcwNzgzNGMwMzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTEwYTJjMjUxYjYxNDE5YWJjY2M2OGU4M2FiMDNiYTUuc2V0Q29udGVudChodG1sXzgyNzEwZjU1MGVmZDRiY2FiNTBlNTg3MDc4MzRjMDMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFjZDY3YTJiMjk2YTQxNTk4OGYxM2E2MDg2Mzk0Y2NiLmJpbmRQb3B1cChwb3B1cF8xMTBhMmMyNTFiNjE0MTlhYmNjYzY4ZTgzYWIwM2JhNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNzJkY2RkYWY0MDU0NjVhOWNmMWRiNjhhZGFlZjhjZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI1ZmEwMzdhNmViNDRhYTA4M2VlMGE0NzYwMDhiNmU4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2YjI3N2JjZjRiODQzZDJiMTRlYTE5OTVjY2UxZWVkID0gJCgnPGRpdiBpZD0iaHRtbF9iNmIyNzdiY2Y0Yjg0M2QyYjE0ZWExOTk1Y2NlMWVlZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBSb3NlbGF3bjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjVmYTAzN2E2ZWI0NGFhMDgzZWUwYTQ3NjAwOGI2ZTguc2V0Q29udGVudChodG1sX2I2YjI3N2JjZjRiODQzZDJiMTRlYTE5OTVjY2UxZWVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE3MmRjZGRhZjQwNTQ2NWE5Y2YxZGI2OGFkYWVmOGNmLmJpbmRQb3B1cChwb3B1cF8yNWZhMDM3YTZlYjQ0YWEwODNlZTBhNDc2MDA4YjZlOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYjM2ZWZlN2EyNjg0ODUyODdmMzI2YTEzOTg5NzEyYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y3OTZjOTJkZjE0NTRjZDU5MWVjZTM2MWYwYjY1ZWYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZlOTNjNDdiMmNlOTRkMjdiZmM2OGU2MGZmZTZlNGIyID0gJCgnPGRpdiBpZD0iaHRtbF82ZTkzYzQ3YjJjZTk0ZDI3YmZjNjhlNjBmZmU2ZTRiMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBGb3Jlc3QgSGlsbCBOb3J0aCwgRm9yZXN0IEhpbGwgV2VzdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjc5NmM5MmRmMTQ1NGNkNTkxZWNlMzYxZjBiNjVlZjMuc2V0Q29udGVudChodG1sXzZlOTNjNDdiMmNlOTRkMjdiZmM2OGU2MGZmZTZlNGIyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RiMzZlZmU3YTI2ODQ4NTI4N2YzMjZhMTM5ODk3MTJhLmJpbmRQb3B1cChwb3B1cF9mNzk2YzkyZGYxNDU0Y2Q1OTFlY2UzNjFmMGI2NWVmMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYWZkOTFhNDEzODA0NGY3OGM2OWNhNjQyZjZlOTVhZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JjYmI0N2FmZWM1NDRlMjViZmY4MmU4NDEzNmZlZjhjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I4NDg1ZGNjZTI3NzQ4ODQ5Mzc4MWNjNWExYjNiODZiID0gJCgnPGRpdiBpZD0iaHRtbF9iODQ4NWRjY2UyNzc0ODg0OTM3ODFjYzVhMWIzYjg2YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBUaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmNiYjQ3YWZlYzU0NGUyNWJmZjgyZTg0MTM2ZmVmOGMuc2V0Q29udGVudChodG1sX2I4NDg1ZGNjZTI3NzQ4ODQ5Mzc4MWNjNWExYjNiODZiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJhZmQ5MWE0MTM4MDQ0Zjc4YzY5Y2E2NDJmNmU5NWFmLmJpbmRQb3B1cChwb3B1cF9iY2JiNDdhZmVjNTQ0ZTI1YmZmODJlODQxMzZmZWY4Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZmM5OGVjNDcyYmY0MDQ4OWNjMDkxYmUwMDQyYTcwOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmIwZmUwM2JhNWIzNDQ1OThjNWRjOGI3Yzc3ZDczZTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjM5NGM0N2RhYTJiNDg0Yzg4YWE2ZGIyYjU4ZDI1MTAgPSAkKCc8ZGl2IGlkPSJodG1sX2YzOTRjNDdkYWEyYjQ4NGM4OGFhNmRiMmI1OGQyNTEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBIYXJib3JkLCBVbml2ZXJzaXR5IG9mIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMGZlMDNiYTViMzQ0NTk4YzVkYzhiN2M3N2Q3M2U1LnNldENvbnRlbnQoaHRtbF9mMzk0YzQ3ZGFhMmI0ODRjODhhYTZkYjJiNThkMjUxMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZmM5OGVjNDcyYmY0MDQ4OWNjMDkxYmUwMDQyYTcwOS5iaW5kUG9wdXAocG9wdXBfZmIwZmUwM2JhNWIzNDQ1OThjNWRjOGI3Yzc3ZDczZTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmFlNDNhNDQ4NzEyNGJkNWE1YzFiODUzYTc0ZTQyM2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlNzg1OTY1N2UyODQ1NDM5NTEyN2RkNmRhNmM0OGJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY0NjVjMDE5ZWI5NjRjMzY4N2E4MmZiNGU5YmNmYjRlID0gJCgnPGRpdiBpZD0iaHRtbF82NDY1YzAxOWViOTY0YzM2ODdhODJmYjRlOWJjZmI0ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogQ2hpbmF0b3duLCBHcmFuZ2UgUGFyaywgS2Vuc2luZ3RvbiBNYXJrZXQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJlNzg1OTY1N2UyODQ1NDM5NTEyN2RkNmRhNmM0OGJlLnNldENvbnRlbnQoaHRtbF82NDY1YzAxOWViOTY0YzM2ODdhODJmYjRlOWJjZmI0ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYWU0M2E0NDg3MTI0YmQ1YTVjMWI4NTNhNzRlNDIzYS5iaW5kUG9wdXAocG9wdXBfMmU3ODU5NjU3ZTI4NDU0Mzk1MTI3ZGQ2ZGE2YzQ4YmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDMzNzU2YWUyNDY0NGRjMGI4MjRiYWNmY2E2NmM4NWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNlYTkxZTc1OTViZjRhMWRiNTA3MTE4Y2I0YjBkMDFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJmMTkwOTA4OTBlZjQ0YzJhNjgwOGM2NjcwN2M3NmFiID0gJCgnPGRpdiBpZD0iaHRtbF8yZjE5MDkwODkwZWY0NGMyYTY4MDhjNjY3MDdjNzZhYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogQ04gVG93ZXIsIEJhdGh1cnN0IFF1YXksIElzbGFuZCBhaXJwb3J0LCBIYXJib3VyZnJvbnQgV2VzdCwgS2luZyBhbmQgU3BhZGluYSwgUmFpbHdheSBMYW5kcywgU291dGggTmlhZ2FyYTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2VhOTFlNzU5NWJmNGExZGI1MDcxMThjYjRiMGQwMWUuc2V0Q29udGVudChodG1sXzJmMTkwOTA4OTBlZjQ0YzJhNjgwOGM2NjcwN2M3NmFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QzMzc1NmFlMjQ2NDRkYzBiODI0YmFjZmNhNjZjODViLmJpbmRQb3B1cChwb3B1cF8zZWE5MWU3NTk1YmY0YTFkYjUwNzExOGNiNGIwZDAxZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MDNhOWM5NjI5YTM0YTZhOTY5YjYyYjNjZDdhMGQ4MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NjQzNTIsLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JiMzk2M2FkNWZlZTQzMmFhZmIzYzQ4MzM5ZTY5OGZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ExZjM3NGJmZGQyZTRmMzU5OTI3MDBkYzU0MmQ2NjdjID0gJCgnPGRpdiBpZD0iaHRtbF9hMWYzNzRiZmRkMmU0ZjM1OTkyNzAwZGM1NDJkNjY3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogU3RuIEEgUE8gQm94ZXMgMjUgVGhlIEVzcGxhbmFkZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmIzOTYzYWQ1ZmVlNDMyYWFmYjNjNDgzMzllNjk4ZmIuc2V0Q29udGVudChodG1sX2ExZjM3NGJmZGQyZTRmMzU5OTI3MDBkYzU0MmQ2NjdjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwM2E5Yzk2MjlhMzRhNmE5NjliNjJiM2NkN2EwZDgzLmJpbmRQb3B1cChwb3B1cF9iYjM5NjNhZDVmZWU0MzJhYWZiM2M0ODMzOWU2OThmYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNmVkOTQ4YjMzZDA0M2I3OGM5NmRmYThjMjRiYTI5ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTk1MmNhNDQ3OWNlNDIwOTk2YmI0NTZhYzk1MjkwZjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWJiZWRhZjE1ZjQ3NGVlZjljZjJlMWRiYWJmM2I5OTUgPSAkKCc8ZGl2IGlkPSJodG1sXzFiYmVkYWYxNWY0NzRlZWY5Y2YyZTFkYmFiZjNiOTk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBGaXJzdCBDYW5hZGlhbiBQbGFjZSwgVW5kZXJncm91bmQgY2l0eTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTk1MmNhNDQ3OWNlNDIwOTk2YmI0NTZhYzk1MjkwZjAuc2V0Q29udGVudChodG1sXzFiYmVkYWYxNWY0NzRlZWY5Y2YyZTFkYmFiZjNiOTk1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q2ZWQ5NDhiMzNkMDQzYjc4Yzk2ZGZhOGMyNGJhMjllLmJpbmRQb3B1cChwb3B1cF9hOTUyY2E0NDc5Y2U0MjA5OTZiYjQ1NmFjOTUyOTBmMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYTc1YzgyY2RkNjA0NzQ4YmYyZWI0ZWIwZjdkNGU4OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZmQzMmM3NWIzNGE0ZTBiOTdkYjE4NDI4MTEyY2M0MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYjNhYzlhMDBmZjY0YmE5YWM3OGMzMTcwZTU3MmFlNCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGIzYWM5YTAwZmY2NGJhOWFjNzhjMzE3MGU1NzJhZTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENocmlzdGllPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZmQzMmM3NWIzNGE0ZTBiOTdkYjE4NDI4MTEyY2M0Mi5zZXRDb250ZW50KGh0bWxfZGIzYWM5YTAwZmY2NGJhOWFjNzhjMzE3MGU1NzJhZTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGE3NWM4MmNkZDYwNDc0OGJmMmViNGViMGY3ZDRlODguYmluZFBvcHVwKHBvcHVwXzdmZDMyYzc1YjM0YTRlMGI5N2RiMTg0MjgxMTJjYzQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBlMzljMDhjYjVjZDRiMDhhYTAxMzg3OTY4MjhjZDJkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDM4NTNiZGQ3NDM4NDM3OGI0Y2VmM2Y1NGEyNGM1MDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmE4MTNiMGQxMGQ4NGU0ZmJhYzZhMjNkZTQ2Yjc5ZWMgPSAkKCc8ZGl2IGlkPSJodG1sXzZhODEzYjBkMTBkODRlNGZiYWM2YTIzZGU0NmI3OWVjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG86IERvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW48L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QzODUzYmRkNzQzODQzNzhiNGNlZjNmNTRhMjRjNTAzLnNldENvbnRlbnQoaHRtbF82YTgxM2IwZDEwZDg0ZTRmYmFjNmEyM2RlNDZiNzllYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZTM5YzA4Y2I1Y2Q0YjA4YWEwMTM4Nzk2ODI4Y2QyZC5iaW5kUG9wdXAocG9wdXBfZDM4NTNiZGQ3NDM4NDM3OGI0Y2VmM2Y1NGEyNGM1MDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTZiZGQ1NjdhMmE1NDk0ODk1YWM3YzkwMWUzOTM4YjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDc5MjY3MDAwMDAwMDYsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDg4OWQ1NmY4OGM5NDFhYjllNDhmN2ViOGQ0NGQ1ZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2ZlOTRhOGQxZTUyNDg0ZGE4MzBjOTIxYmRkYTg0MWIgPSAkKCc8ZGl2IGlkPSJodG1sXzdmZTk0YThkMWU1MjQ4NGRhODMwYzkyMWJkZGE4NDFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG86IExpdHRsZSBQb3J0dWdhbCwgVHJpbml0eTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDg4OWQ1NmY4OGM5NDFhYjllNDhmN2ViOGQ0NGQ1ZjYuc2V0Q29udGVudChodG1sXzdmZTk0YThkMWU1MjQ4NGRhODMwYzkyMWJkZGE4NDFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU2YmRkNTY3YTJhNTQ5NDg5NWFjN2M5MDFlMzkzOGIxLmJpbmRQb3B1cChwb3B1cF9kODg5ZDU2Zjg4Yzk0MWFiOWU0OGY3ZWI4ZDQ0ZDVmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZTg0Njg5ODA5MTQ0MDdkODc5YjNiNmEwZWJiMmQ0YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjg0NzIsLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwZWNlZTJkZDAyYzQyZWQ4MWU4YzI1NmE3NTAxM2NiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MxZDA2NGQyZDQ3MzQyYWU4YzQyNzkxMWVhNTEzNmE4ID0gJCgnPGRpdiBpZD0iaHRtbF9jMWQwNjRkMmQ0NzM0MmFlOGM0Mjc5MTFlYTUxMzZhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvOiBCcm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTBlY2VlMmRkMDJjNDJlZDgxZThjMjU2YTc1MDEzY2Iuc2V0Q29udGVudChodG1sX2MxZDA2NGQyZDQ3MzQyYWU4YzQyNzkxMWVhNTEzNmE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlODQ2ODk4MDkxNDQwN2Q4NzliM2I2YTBlYmIyZDRjLmJpbmRQb3B1cChwb3B1cF9hMGVjZWUyZGQwMmM0MmVkODFlOGMyNTZhNzUwMTNjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MjgyZDk1ODE0Y2E0M2QzYWNlMDZiOTdhZjc4ZDA5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NlOThjM2YzYjhjMjQ5ZDNiZGNhYWE3Yjk3MTk4ODY1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNmNzJiYWRlMjhkODRkNTViM2Y4YTk0NDk4Njk2MTk2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyMzYzYjkyNGI2MTRmMDdiY2ExZjg4MTI1ZjE2YzAzID0gJCgnPGRpdiBpZD0iaHRtbF82MjM2M2I5MjRiNjE0ZjA3YmNhMWY4ODEyNWYxNmMwMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvOiBIaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2Y3MmJhZGUyOGQ4NGQ1NWIzZjhhOTQ0OTg2OTYxOTYuc2V0Q29udGVudChodG1sXzYyMzYzYjkyNGI2MTRmMDdiY2ExZjg4MTI1ZjE2YzAzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyODJkOTU4MTRjYTQzZDNhY2UwNmI5N2FmNzhkMDkyLmJpbmRQb3B1cChwb3B1cF8zZjcyYmFkZTI4ZDg0ZDU1YjNmOGE5NDQ5ODY5NjE5Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZGQwZjJjZTMzZGM0MzI1OGZhODU5NjYzNGJiMWQ2ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODk1OTcsLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jOTBjNWUwZmFmNzc0M2RhYjg4Mjk0Y2JhZWYwZTQ3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZmMyYTdmZTk3Y2Q0ODFmYjhmNzNlYjYzYjZmMjhlNCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2ZjMmE3ZmU5N2NkNDgxZmI4ZjczZWI2M2I2ZjI4ZTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bzogUGFya2RhbGUsIFJvbmNlc3ZhbGxlczwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzkwYzVlMGZhZjc3NDNkYWI4ODI5NGNiYWVmMGU0N2Euc2V0Q29udGVudChodG1sXzNmYzJhN2ZlOTdjZDQ4MWZiOGY3M2ViNjNiNmYyOGU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NkZDBmMmNlMzNkYzQzMjU4ZmE4NTk2NjM0YmIxZDZlLmJpbmRQb3B1cChwb3B1cF9jOTBjNWUwZmFmNzc0M2RhYjg4Mjk0Y2JhZWYwZTQ3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OGFiNzUyZDY0NDY0MDlmYTEwYWRiZmM0N2I3ZTFkYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTU3MDYsLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2U5OGMzZjNiOGMyNDlkM2JkY2FhYTdiOTcxOTg4NjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjU2YzVkYTA3ZDVkNGNlODhkODU4ZmNkNTkzODA5MmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2U5YzRmYmVmNjcxNGFjY2FhNTc3YjI1YTYwM2Y3Y2QgPSAkKCc8ZGl2IGlkPSJodG1sXzNlOWM0ZmJlZjY3MTRhY2NhYTU3N2IyNWE2MDNmN2NkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG86IFJ1bm55bWVkZSwgU3dhbnNlYTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjU2YzVkYTA3ZDVkNGNlODhkODU4ZmNkNTkzODA5MmUuc2V0Q29udGVudChodG1sXzNlOWM0ZmJlZjY3MTRhY2NhYTU3N2IyNWE2MDNmN2NkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4YWI3NTJkNjQ0NjQwOWZhMTBhZGJmYzQ3YjdlMWRiLmJpbmRQb3B1cChwb3B1cF8yNTZjNWRhMDdkNWQ0Y2U4OGQ4NThmY2Q1OTM4MDkyZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNDA5MzU2ZDIxMWY0ZWEzOGVkNGM0MDY1YTY5YjRkMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Mjc0MzksLTc5LjMyMTU1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jZTk4YzNmM2I4YzI0OWQzYmRjYWFhN2I5NzE5ODg2NSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MzkzN2Q0MTE1M2E0Y2I5YmM5ODIzMTUwY2FhNjRkNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kN2VhZWEwOGYyMzQ0MmQxYTIzZGVjNDAwY2MzYzVlNCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDdlYWVhMDhmMjM0NDJkMWEyM2RlYzQwMGNjM2M1ZTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250bzogQnVzaW5lc3MgUmVwbHkgTWFpbCBQcm9jZXNzaW5nIENlbnRyZSA5NjkgRWFzdGVybjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjM5MzdkNDExNTNhNGNiOWJjOTgyMzE1MGNhYTY0ZDUuc2V0Q29udGVudChodG1sX2Q3ZWFlYTA4ZjIzNDQyZDFhMjNkZWM0MDBjYzNjNWU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U0MDkzNTZkMjExZjRlYTM4ZWQ0YzQwNjVhNjliNGQzLmJpbmRQb3B1cChwb3B1cF82MzkzN2Q0MTE1M2E0Y2I5YmM5ODIzMTUwY2FhNjRkNSk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
CLIENT_ID = 'FROUNNKNE2NFNJ3DPMSVCFHY5322WYELMEGST2E22VVQCDCN' # your Foursquare ID
CLIENT_SECRET = 'Z30WRUBMICAZU4NBL4UOTJTKB3LWTDVTRAXJU3AFIX3ONRA0' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_SECRET:' + CLIENT_SECRET)
print('CLIENT_ID: ' + CLIENT_ID)
```

    Your credentails:
    CLIENT_SECRET:Z30WRUBMICAZU4NBL4UOTJTKB3LWTDVTRAXJU3AFIX3ONRA0
    CLIENT_ID: FROUNNKNE2NFNJ3DPMSVCFHY5322WYELMEGST2E22VVQCDCN


##### We will explore the neighborhoods using the Foursquare API in the following cells


```python
#creating a function to get the  near by venues of a given neighborhood using the Foursquare API

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
#Obtaining venues within 500m of each neighborhood with a limit of 100 venues per neighborhood
LIMIT=100
toronto_venues = getNearbyVenues(names=df_toronto['Neighborhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude']
                                  )
toronto_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Manor Ravine</td>
      <td>43.676821</td>
      <td>-79.293942</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>The Big Carrot Natural Food Market</td>
      <td>43.678879</td>
      <td>-79.297734</td>
      <td>Health Food Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Grover Pub and Grub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Upper Beaches</td>
      <td>43.680563</td>
      <td>-79.292869</td>
      <td>Neighborhood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Pantheon</td>
      <td>43.677621</td>
      <td>-79.351434</td>
      <td>Greek Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



##### We will check how many venues were obtained per neighborhood below


```python
toronto_venues.groupby('Neighborhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adelaide, King, Richmond</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Berczy Park</th>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
    </tr>
    <tr>
      <th>Brockton, Exhibition Place, Parkdale Village</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Business Reply Mail Processing Centre 969 Eastern</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>CN Tower, Bathurst Quay, Island airport, Harbourfront West, King and Spadina, Railway Lands, South Niagara</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Cabbagetown, St. James Town</th>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Chinatown, Grange Park, Kensington Market</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>89</td>
      <td>89</td>
      <td>89</td>
      <td>89</td>
      <td>89</td>
      <td>89</td>
    </tr>
    <tr>
      <th>Commerce Court, Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Davisville</th>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Davisville North</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Design Exchange, Toronto Dominion Centre</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Dovercourt Village, Dufferin</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Forest Hill North, Forest Hill West</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Harbord, University of Toronto</th>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>Harbourfront</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Harbourfront East, Toronto Islands, Union Station</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>High Park, The Junction South</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
    <tr>
      <th>Lawrence Park</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Little Portugal, Trinity</th>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Moore Park, Summerhill East</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>North Toronto West</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Parkdale, Roncesvalles</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Roselawn</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Runnymede, Swansea</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Ryerson, Garden District</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
    </tr>
    <tr>
      <th>Studio District</th>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>The Annex, North Midtown, Yorkville</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>The Beaches</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>The Beaches West, India Bazaar</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>The Danforth West, Riverdale</th>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Checking how many unique venue categories were obtained
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
```

    There are 234 uniques categories.


##### We will analyse each neigborhood in the cells below


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhoods'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhoods</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Board Shop</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Costume Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Home Service</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Malay Restaurant</th>
      <th>Market</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Movie Theater</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recording Studio</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tapas Restaurant</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Danforth West, Riverdale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category


```python
#Grouping the 
toronto_grouped = toronto_onehot.groupby('Neighborhoods').mean().reset_index()
toronto_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhoods</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Board Shop</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Costume Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Home Service</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Malay Restaurant</th>
      <th>Market</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Movie Theater</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recording Studio</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tapas Restaurant</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide, King, Richmond</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.01</td>
      <td>0.0000</td>
      <td>0.01</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.070000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.030000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.030000</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.035088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.035088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.035088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>0.035088</td>
      <td>0.070175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.035088</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.017544</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.035088</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.035088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017544</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.095238</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.095238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.095238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>0.0</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.125</td>
      <td>0.1875</td>
      <td>0.0625</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062500</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0625</td>
      <td>0.00</td>
      <td>0.0625</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Let's get the neighborhoods with their top 5 most common venues


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python

num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhoods']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhoods'] = toronto_grouped['Neighborhoods']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.rename(columns={'Neighborhoods':'Neighborhood'},inplace=True)
neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide, King, Richmond</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Beer Bar</td>
      <td>Cheese Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Breakfast Spot</td>
      <td>Grocery Store</td>
      <td>Intersection</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>Light Rail Station</td>
      <td>Pizza Place</td>
      <td>Auto Workshop</td>
      <td>Comic Shop</td>
      <td>Recording Studio</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Plane</td>
      <td>Boutique</td>
      <td>Bar</td>
    </tr>
  </tbody>
</table>
</div>



##### Clustering the neighborhoods using Kmeans 


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhoods', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
           4, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)



##### Creating a dataframe with cluster labels and the top 5 venues


```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>0</td>
      <td>Neighborhood</td>
      <td>Health Food Store</td>
      <td>Pub</td>
      <td>Trail</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Italian Restaurant</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>0</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Brewery</td>
      <td>Steakhouse</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>4</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Bus Line</td>
      <td>Yoga Studio</td>
      <td>Diner</td>
    </tr>
  </tbody>
</table>
</div>



##### We will display the neighborhoods along with their  cluster labels and colours on a map below


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, bor, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'],toronto_merged['Borough'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(bor)+': ' +str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjY3MjYyMTg0MjEwNTIsLTc5LjM4OTg4MzIzNDIxMDUzXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDEyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl84ZTU5Njk1Y2JkYzQ0ZGNmYTdlZjNhMjM1OTg1MTExNSA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDM4Y2UwZDUxOTBkNDBhYTlhNGJhMmQ3MGEyMDcyNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNjk2NGJiYjJiNzE0N2U3YTNlMWQyYzA4NDU5ZTc0NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ODI4NTkyYmNiZjk0ZGIyOGJmMjAxZWQwMjNlMmM0NiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjgyODU5MmJjYmY5NGRiMjhiZjIwMWVkMDIzZTJjNDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250bzogVGhlIEJlYWNoZXMgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNjk2NGJiYjJiNzE0N2U3YTNlMWQyYzA4NDU5ZTc0Ni5zZXRDb250ZW50KGh0bWxfNjgyODU5MmJjYmY5NGRiMjhiZjIwMWVkMDIzZTJjNDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDM4Y2UwZDUxOTBkNDBhYTlhNGJhMmQ3MGEyMDcyNjguYmluZFBvcHVwKHBvcHVwXzM2OTY0YmJiMmI3MTQ3ZTdhM2UxZDJjMDg0NTllNzQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc0ODhhNjhjMGY0ZTQwZGZhM2ZjYjY4Y2M3NDc4OTUzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBmZWFhZDI5YmFkYzQxM2NhNjk3NWRjODE0MzVhNGQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJlNjQ3NzI0NDk2MDQzYjJiOGE2Y2RhYjMxZjY3MDRhID0gJCgnPGRpdiBpZD0iaHRtbF8yZTY0NzcyNDQ5NjA0M2IyYjhhNmNkYWIzMWY2NzA0YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvOiBUaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGZlYWFkMjliYWRjNDEzY2E2OTc1ZGM4MTQzNWE0ZDEuc2V0Q29udGVudChodG1sXzJlNjQ3NzI0NDk2MDQzYjJiOGE2Y2RhYjMxZjY3MDRhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc0ODhhNjhjMGY0ZTQwZGZhM2ZjYjY4Y2M3NDc4OTUzLmJpbmRQb3B1cChwb3B1cF8wZmVhYWQyOWJhZGM0MTNjYTY5NzVkYzgxNDM1YTRkMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NjZkYTFkODE5NzU0Yjc4OTFjZjBiNDhlNmJhNTVlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ1OGJhYjIyMGFiYzRmY2ViMzk0MDI5MjBmYTMyYTk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ5NzAzNWQ1Y2VmMDRiYTQ5YWI1OWEzZDMwMmNhNzNhID0gJCgnPGRpdiBpZD0iaHRtbF80OTcwMzVkNWNlZjA0YmE0OWFiNTlhM2QzMDJjYTczYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvOiBUaGUgQmVhY2hlcyBXZXN0LCBJbmRpYSBCYXphYXIgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NThiYWIyMjBhYmM0ZmNlYjM5NDAyOTIwZmEzMmE5Ny5zZXRDb250ZW50KGh0bWxfNDk3MDM1ZDVjZWYwNGJhNDlhYjU5YTNkMzAyY2E3M2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDY2ZGExZDgxOTc1NGI3ODkxY2YwYjQ4ZTZiYTU1ZTEuYmluZFBvcHVwKHBvcHVwXzQ1OGJhYjIyMGFiYzRmY2ViMzk0MDI5MjBmYTMyYTk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y3MDVjOGU0Y2EwODQwZWNiZmVhMjliNzY5NTZlY2Y5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA0YjMxMDQxMDU3YzQ4NTRhYzA2NjRmYjUwZDM2MDA2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgyNzllYWQ3Y2Q0MDQ3MjJiN2E0MDdkNTk0YjdjMDc4ID0gJCgnPGRpdiBpZD0iaHRtbF84Mjc5ZWFkN2NkNDA0NzIyYjdhNDA3ZDU5NGI3YzA3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvOiBTdHVkaW8gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNGIzMTA0MTA1N2M0ODU0YWMwNjY0ZmI1MGQzNjAwNi5zZXRDb250ZW50KGh0bWxfODI3OWVhZDdjZDQwNDcyMmI3YTQwN2Q1OTRiN2MwNzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjcwNWM4ZTRjYTA4NDBlY2JmZWEyOWI3Njk1NmVjZjkuYmluZFBvcHVwKHBvcHVwXzA0YjMxMDQxMDU3YzQ4NTRhYzA2NjRmYjUwZDM2MDA2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzViNzUxMTE4ODE5ZTQ5YjRhYTMwNDM5Y2ZjY2Y1ZGIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ODM3NzUxYTUxYTA0MjQ1OTI4MTVkMTBiMmIyNTI2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZTYwNWFhYmQ0N2Q0NDU5OWQ2Y2VjMmI3ZmY4OTMxMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNGU2MDVhYWJkNDdkNDQ1OTlkNmNlYzJiN2ZmODkzMTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bzogTGF3cmVuY2UgUGFyayBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4Mzc3NTFhNTFhMDQyNDU5MjgxNWQxMGIyYjI1MjY4LnNldENvbnRlbnQoaHRtbF80ZTYwNWFhYmQ0N2Q0NDU5OWQ2Y2VjMmI3ZmY4OTMxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81Yjc1MTExODgxOWU0OWI0YWEzMDQzOWNmY2NmNWRiMS5iaW5kUG9wdXAocG9wdXBfNzgzNzc1MWE1MWEwNDI0NTkyODE1ZDEwYjJiMjUyNjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDZhNjlkMjc3YjRjNDY1NmE1YzA0ZGZiZDk3MGU1YzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YyNWI2OTI0ZGEyOTQ2YWNiMWI3YTRhOTU1ZTM3ZDBjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA1MzNlOTkxY2M2ZTRjN2JiNDFmMjc2OWExOGIyZDhjID0gJCgnPGRpdiBpZD0iaHRtbF8wNTMzZTk5MWNjNmU0YzdiYjQxZjI3NjlhMThiMmQ4YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBEYXZpc3ZpbGxlIE5vcnRoIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjI1YjY5MjRkYTI5NDZhY2IxYjdhNGE5NTVlMzdkMGMuc2V0Q29udGVudChodG1sXzA1MzNlOTkxY2M2ZTRjN2JiNDFmMjc2OWExOGIyZDhjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2YTY5ZDI3N2I0YzQ2NTZhNWMwNGRmYmQ5NzBlNWMxLmJpbmRQb3B1cChwb3B1cF9mMjViNjkyNGRhMjk0NmFjYjFiN2E0YTk1NWUzN2QwYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81OGQ0MDU3NGQxZDM0NjhiODI4YTcyMGY2Y2Q5NGFhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNTM4MzQsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE3YWY5NDk1MzgyYjQwNzlhMTIwYzIxYjM4ZGQ4MzMyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhhZDVhNTMxMzEzNTQxZmZiMTAwMjM2NmYyZTY0M2E3ID0gJCgnPGRpdiBpZD0iaHRtbF84YWQ1YTUzMTMxMzU0MWZmYjEwMDIzNjZmMmU2NDNhNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBOb3J0aCBUb3JvbnRvIFdlc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xN2FmOTQ5NTM4MmI0MDc5YTEyMGMyMWIzOGRkODMzMi5zZXRDb250ZW50KGh0bWxfOGFkNWE1MzEzMTM1NDFmZmIxMDAyMzY2ZjJlNjQzYTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNThkNDA1NzRkMWQzNDY4YjgyOGE3MjBmNmNkOTRhYWQuYmluZFBvcHVwKHBvcHVwXzE3YWY5NDk1MzgyYjQwNzlhMTIwYzIxYjM4ZGQ4MzMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q1Y2MyY2ZiZTc4ZDQyMWI5NTZjZmJiMzYxMTI4OGE1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MjcxYjIxMDRmMGM0YThmYmZiNjIyZWI1OWNmMmFjYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NzQzMjBkMDhhM2Y0YjI2OTI5OWVhZjQ4YmI4NzhlZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzc0MzIwZDA4YTNmNGIyNjkyOTllYWY0OGJiODc4ZWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bzogRGF2aXN2aWxsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcyNzFiMjEwNGYwYzRhOGZiZmI2MjJlYjU5Y2YyYWNjLnNldENvbnRlbnQoaHRtbF83NzQzMjBkMDhhM2Y0YjI2OTI5OWVhZjQ4YmI4NzhlZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNWNjMmNmYmU3OGQ0MjFiOTU2Y2ZiYjM2MTEyODhhNS5iaW5kUG9wdXAocG9wdXBfNzI3MWIyMTA0ZjBjNGE4ZmJmYjYyMmViNTljZjJhY2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmVhNDI4ODZhMGZkNDdmNDgxMWEyN2M5YmE2MWJmMTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMjRmYWVlNDc1Yzk0MDBiODVkY2UzMGI3NjhmMjc2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZjAzMmRiZWQzZjE0ZmRlYjI3Yjg2MTI3MjNiYzY0OCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmYwMzJkYmVkM2YxNGZkZWIyN2I4NjEyNzIzYmM2NDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bzogTW9vcmUgUGFyaywgU3VtbWVyaGlsbCBFYXN0IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjI0ZmFlZTQ3NWM5NDAwYjg1ZGNlMzBiNzY4ZjI3NmQuc2V0Q29udGVudChodG1sX2JmMDMyZGJlZDNmMTRmZGViMjdiODYxMjcyM2JjNjQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JlYTQyODg2YTBmZDQ3ZjQ4MTFhMjdjOWJhNjFiZjEyLmJpbmRQb3B1cChwb3B1cF8yMjRmYWVlNDc1Yzk0MDBiODVkY2UzMGI3NjhmMjc2ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMTM2NDViZDEwNzg0MzY2YTQ1ZTZlZGE2NDU4ZjI2ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1ZWFhOWVhN2YxMjQ2MGJhMTMwNGJkNmM3MWVkYTE4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhmYTMyZDcwZWUwZDQzYmE5NTc0OGJjNDI5NzI1NjdhID0gJCgnPGRpdiBpZD0iaHRtbF84ZmEzMmQ3MGVlMGQ0M2JhOTU3NDhiYzQyOTcyNTY3YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBEZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIFN1bW1lcmhpbGwgV2VzdCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U1ZWFhOWVhN2YxMjQ2MGJhMTMwNGJkNmM3MWVkYTE4LnNldENvbnRlbnQoaHRtbF84ZmEzMmQ3MGVlMGQ0M2JhOTU3NDhiYzQyOTcyNTY3YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMTM2NDViZDEwNzg0MzY2YTQ1ZTZlZGE2NDU4ZjI2ZS5iaW5kUG9wdXAocG9wdXBfZTVlYWE5ZWE3ZjEyNDYwYmExMzA0YmQ2YzcxZWRhMTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2M2ZTI3M2I3MjU5NGU4MmEyOGY5N2UwODdkY2I3NDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ODkxMWNlYTNiY2E0ZTIyOWZlOTc5NjY3ZjM0ZGUyNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZjUxNTBhZmZkMzE0YTcxOWE1NTBiYjlhOGI0NzU3YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGY1MTUwYWZmZDMxNGE3MTlhNTUwYmI5YThiNDc1N2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IFJvc2VkYWxlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjg5MTFjZWEzYmNhNGUyMjlmZTk3OTY2N2YzNGRlMjQuc2V0Q29udGVudChodG1sX2RmNTE1MGFmZmQzMTRhNzE5YTU1MGJiOWE4YjQ3NTdiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NjNmUyNzNiNzI1OTRlODJhMjhmOTdlMDg3ZGNiNzQxLmJpbmRQb3B1cChwb3B1cF82ODkxMWNlYTNiY2E0ZTIyOWZlOTc5NjY3ZjM0ZGUyNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZTA2ZTllZmFkYWY0YTRjYmViMDkyMjY4ZTkxZTkwZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MzZlYzY2MDZjMTU0N2JlOWRjMTk3ZmZiOGRiNmVmYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMjk2YjhmOGYzMjI0MWI4OGE5NmNkZjgxNTVkOWUwNSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzI5NmI4ZjhmMzIyNDFiODhhOTZjZGY4MTU1ZDllMDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYzNmVjNjYwNmMxNTQ3YmU5ZGMxOTdmZmI4ZGI2ZWZjLnNldENvbnRlbnQoaHRtbF9jMjk2YjhmOGYzMjI0MWI4OGE5NmNkZjgxNTVkOWUwNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZTA2ZTllZmFkYWY0YTRjYmViMDkyMjY4ZTkxZTkwZC5iaW5kUG9wdXAocG9wdXBfNjM2ZWM2NjA2YzE1NDdiZTlkYzE5N2ZmYjhkYjZlZmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzlhMDRkYzc4YTE5NDM0YjkzMmRmZDFmZjQ4NDNkOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNmU4NTQ3MDVmNWM0NTMwYmYwM2UzODMwYjBlMDk4NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NDVmM2U1NmY5ZWU0OTMyYjM2ZjIzYjkzZjRiNjI5YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjQ1ZjNlNTZmOWVlNDkzMmIzNmYyM2I5M2Y0YjYyOWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENodXJjaCBhbmQgV2VsbGVzbGV5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTZlODU0NzA1ZjVjNDUzMGJmMDNlMzgzMGIwZTA5ODQuc2V0Q29udGVudChodG1sXzY0NWYzZTU2ZjllZTQ5MzJiMzZmMjNiOTNmNGI2MjlhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM5YTA0ZGM3OGExOTQzNGI5MzJkZmQxZmY0ODQzZDljLmJpbmRQb3B1cChwb3B1cF8xNmU4NTQ3MDVmNWM0NTMwYmYwM2UzODMwYjBlMDk4NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYThmYjA5NjM5OWI0NWZiYjI0YjE5MWFhNzNlZGRhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjNhZWVkODcxMjQ5NDU5ZGE0YzYyMzQxNTM1NTBjMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2E4NmIyOWJmNDdjNDMzMGI2NDU5M2Y1ODg1ODE5ODAgPSAkKCc8ZGl2IGlkPSJodG1sX2NhODZiMjliZjQ3YzQzMzBiNjQ1OTNmNTg4NTgxOTgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBIYXJib3VyZnJvbnQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yM2FlZWQ4NzEyNDk0NTlkYTRjNjIzNDE1MzU1MGMwMy5zZXRDb250ZW50KGh0bWxfY2E4NmIyOWJmNDdjNDMzMGI2NDU5M2Y1ODg1ODE5ODApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGE4ZmIwOTYzOTliNDVmYmIyNGIxOTFhYTczZWRkYTguYmluZFBvcHVwKHBvcHVwXzIzYWVlZDg3MTI0OTQ1OWRhNGM2MjM0MTUzNTUwYzAzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IzYjUwOWQ2MDc0MjRlMGNiNWI3Mzc5Y2MwNGMwMjEzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjMyNTQ0ZTEwM2E5NGZlY2FjY2NmN2FjMzA3ZmI2OGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2YwMTQxMzI2YTY3NDMwMDgyYTMxYjRiYjRjNzRkYTAgPSAkKCc8ZGl2IGlkPSJodG1sXzdmMDE0MTMyNmE2NzQzMDA4MmEzMWI0YmI0Yzc0ZGEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBSeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMzI1NDRlMTAzYTk0ZmVjYWNjY2Y3YWMzMDdmYjY4ZS5zZXRDb250ZW50KGh0bWxfN2YwMTQxMzI2YTY3NDMwMDgyYTMxYjRiYjRjNzRkYTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjNiNTA5ZDYwNzQyNGUwY2I1YjczNzljYzA0YzAyMTMuYmluZFBvcHVwKHBvcHVwX2IzMjU0NGUxMDNhOTRmZWNhY2NjZjdhYzMwN2ZiNjhlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc5MDAzZTQwMGVjYjQ3MmE4Y2IwOGQ3YjA4YWM2NjFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMGJjMTk4YjlmNmQ0MTBmODIxNjIyN2JlMDkxYjhiNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NDRmYTNkYzNjZjc0YjljYmNhMTljZDc3NGMwYTg3NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTQ0ZmEzZGMzY2Y3NGI5Y2JjYTE5Y2Q3NzRjMGE4NzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IFN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTBiYzE5OGI5ZjZkNDEwZjgyMTYyMjdiZTA5MWI4YjUuc2V0Q29udGVudChodG1sXzU0NGZhM2RjM2NmNzRiOWNiY2ExOWNkNzc0YzBhODc0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc5MDAzZTQwMGVjYjQ3MmE4Y2IwOGQ3YjA4YWM2NjFlLmJpbmRQb3B1cChwb3B1cF8xMGJjMTk4YjlmNmQ0MTBmODIxNjIyN2JlMDkxYjhiNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTBlMWY3MzA1Njg0NTI3OWYyYzRlOThlM2Q1MGUwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82OTkzMGYyZTBhY2Y0ZTc2YWY2NGNhNTNjZWYyNzg2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMmRlZDMyNGJjMWU0NjM4OTc2ZjZmN2Q0NzU3OGMyNiA9ICQoJzxkaXYgaWQ9Imh0bWxfYTJkZWQzMjRiYzFlNDYzODk3NmY2ZjdkNDc1NzhjMjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IEJlcmN6eSBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjk5MzBmMmUwYWNmNGU3NmFmNjRjYTUzY2VmMjc4Njguc2V0Q29udGVudChodG1sX2EyZGVkMzI0YmMxZTQ2Mzg5NzZmNmY3ZDQ3NTc4YzI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk1MGUxZjczMDU2ODQ1Mjc5ZjJjNGU5OGUzZDUwZTAwLmJpbmRQb3B1cChwb3B1cF82OTkzMGYyZTBhY2Y0ZTc2YWY2NGNhNTNjZWYyNzg2OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YmZmMmVhYmYxNTc0YWFiYTBiNDY1N2QyMjBkZDViYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjQyZTA2ZjIyMGJiNDI4NWIzMGMyZjUxNWU5YWI3MDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWNhMjMwNTRjZGFkNDBlYWE3ZWEyM2EyNzAyOWViZjIgPSAkKCc8ZGl2IGlkPSJodG1sX2FjYTIzMDU0Y2RhZDQwZWFhN2VhMjNhMjcwMjllYmYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBDZW50cmFsIEJheSBTdHJlZXQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNDJlMDZmMjIwYmI0Mjg1YjMwYzJmNTE1ZTlhYjcwNy5zZXRDb250ZW50KGh0bWxfYWNhMjMwNTRjZGFkNDBlYWE3ZWEyM2EyNzAyOWViZjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGJmZjJlYWJmMTU3NGFhYmEwYjQ2NTdkMjIwZGQ1YmIuYmluZFBvcHVwKHBvcHVwX2Y0MmUwNmYyMjBiYjQyODViMzBjMmY1MTVlOWFiNzA3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgzOTc0NWUwYmJiMzRlMjRhODViZWU3ZmQ2YjkwODlmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTdkNjVlNTdiMzYzNGFjMjkyNjNkN2YzYWQ3MTliNjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWRiM2M0NGE0MDhlNGY1NmI4YjA4YmIyMzNkM2Q1NzYgPSAkKCc8ZGl2IGlkPSJodG1sXzlkYjNjNDRhNDA4ZTRmNTZiOGIwOGJiMjMzZDNkNTc2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBBZGVsYWlkZSwgS2luZywgUmljaG1vbmQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hN2Q2NWU1N2IzNjM0YWMyOTI2M2Q3ZjNhZDcxOWI2MS5zZXRDb250ZW50KGh0bWxfOWRiM2M0NGE0MDhlNGY1NmI4YjA4YmIyMzNkM2Q1NzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODM5NzQ1ZTBiYmIzNGUyNGE4NWJlZTdmZDZiOTA4OWYuYmluZFBvcHVwKHBvcHVwX2E3ZDY1ZTU3YjM2MzRhYzI5MjYzZDdmM2FkNzE5YjYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdlNmVmYzUwOWQ0MDQ3NDI5YWUxMDkxOWNhMzhhMTA1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTQ5MGM3MDQzOGRhNDVlM2I5MDAwNzZlZDczNjhjZDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTk1MzliYTU3NjVlNDc2MDhjNDU0MzEzYTU4MTI4N2YgPSAkKCc8ZGl2IGlkPSJodG1sX2E5NTM5YmE1NzY1ZTQ3NjA4YzQ1NDMxM2E1ODEyODdmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBIYXJib3VyZnJvbnQgRWFzdCwgVG9yb250byBJc2xhbmRzLCBVbmlvbiBTdGF0aW9uIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTQ5MGM3MDQzOGRhNDVlM2I5MDAwNzZlZDczNjhjZDAuc2V0Q29udGVudChodG1sX2E5NTM5YmE1NzY1ZTQ3NjA4YzQ1NDMxM2E1ODEyODdmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdlNmVmYzUwOWQ0MDQ3NDI5YWUxMDkxOWNhMzhhMTA1LmJpbmRQb3B1cChwb3B1cF9hNDkwYzcwNDM4ZGE0NWUzYjkwMDA3NmVkNzM2OGNkMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNDQwMmIzYjRlNDQ0NDYxYWU1ZGU3MjhhMzIwNTlhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FjOTc3ZjFjZWI2MzRmYmNiMjRhYTk1NjEwZWM5MjZhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU5MjM0N2Y3MWI1MTQ1NGRhZmY3ZTZlMzNiY2EzNjc4ID0gJCgnPGRpdiBpZD0iaHRtbF81OTIzNDdmNzFiNTE0NTRkYWZmN2U2ZTMzYmNhMzY3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogRGVzaWduIEV4Y2hhbmdlLCBUb3JvbnRvIERvbWluaW9uIENlbnRyZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjOTc3ZjFjZWI2MzRmYmNiMjRhYTk1NjEwZWM5MjZhLnNldENvbnRlbnQoaHRtbF81OTIzNDdmNzFiNTE0NTRkYWZmN2U2ZTMzYmNhMzY3OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNDQwMmIzYjRlNDQ0NDYxYWU1ZGU3MjhhMzIwNTlhZS5iaW5kUG9wdXAocG9wdXBfYWM5NzdmMWNlYjYzNGZiY2IyNGFhOTU2MTBlYzkyNmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTYyYjU3MzM3MGYxNDUzNzhmZjBkYWM4M2I2MGU2MzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMjVkMTE1NWNlZDE0MmUyYjU0MDZjODkzNWM1M2JmOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NjUyZmJiNjEwYmY0Y2M1YTdhMmRjYjdiMjA4OWQwZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjY1MmZiYjYxMGJmNGNjNWE3YTJkY2I3YjIwODlkMGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MyNWQxMTU1Y2VkMTQyZTJiNTQwNmM4OTM1YzUzYmY5LnNldENvbnRlbnQoaHRtbF82NjUyZmJiNjEwYmY0Y2M1YTdhMmRjYjdiMjA4OWQwZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNjJiNTczMzcwZjE0NTM3OGZmMGRhYzgzYjYwZTYzOS5iaW5kUG9wdXAocG9wdXBfYzI1ZDExNTVjZWQxNDJlMmI1NDA2Yzg5MzVjNTNiZjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmZkODE3ZThjNDk4NGQ4ZDkwNWFmZmMyZDEyM2UyNDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ODhlNDA4NGM2MGI0MzZiYjA5MzI0ODFhYmIyNTRlOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZDg3MTEwMDNhNTU0M2Y4OGZiZTRiZWUzZjA3MGVlYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNmQ4NzExMDAzYTU1NDNmODhmYmU0YmVlM2YwNzBlZWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bzogUm9zZWxhd24gQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ODhlNDA4NGM2MGI0MzZiYjA5MzI0ODFhYmIyNTRlOC5zZXRDb250ZW50KGh0bWxfNmQ4NzExMDAzYTU1NDNmODhmYmU0YmVlM2YwNzBlZWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmZkODE3ZThjNDk4NGQ4ZDkwNWFmZmMyZDEyM2UyNDMuYmluZFBvcHVwKHBvcHVwXzg4OGU0MDg0YzYwYjQzNmJiMDkzMjQ4MWFiYjI1NGU4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3NTA0NDAzMWRhODRmMjU4YWUyODQyZDdjMjE2ZGIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTUzYzhlNDdiOGFhNDM0NWJkMzU3N2YzMTI1N2VhYWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2QxYWNmMGFlMzNkNDc4YjgwOWFjZDZkMmU0NzA3NmIgPSAkKCc8ZGl2IGlkPSJodG1sX2NkMWFjZjBhZTMzZDQ3OGI4MDlhY2Q2ZDJlNDcwNzZiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG86IEZvcmVzdCBIaWxsIE5vcnRoLCBGb3Jlc3QgSGlsbCBXZXN0IENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTUzYzhlNDdiOGFhNDM0NWJkMzU3N2YzMTI1N2VhYWQuc2V0Q29udGVudChodG1sX2NkMWFjZjBhZTMzZDQ3OGI4MDlhY2Q2ZDJlNDcwNzZiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk3NTA0NDAzMWRhODRmMjU4YWUyODQyZDdjMjE2ZGIzLmJpbmRQb3B1cChwb3B1cF85NTNjOGU0N2I4YWE0MzQ1YmQzNTc3ZjMxMjU3ZWFhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MDM3YjU1Mjk2NDI0YjQyOGZkYTYyM2ZlNzg0M2ZjYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUxOTg2YzRiYzA0NTRlNjc5NzJjYjg0NTE3OTBiMjdkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkxOGM0NGQ1ODk1ZTQ1MDNhYzJiNDUxZDYzNzEwY2Y0ID0gJCgnPGRpdiBpZD0iaHRtbF85MThjNDRkNTg5NWU0NTAzYWMyYjQ1MWQ2MzcxMGNmNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvOiBUaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUxOTg2YzRiYzA0NTRlNjc5NzJjYjg0NTE3OTBiMjdkLnNldENvbnRlbnQoaHRtbF85MThjNDRkNTg5NWU0NTAzYWMyYjQ1MWQ2MzcxMGNmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MDM3YjU1Mjk2NDI0YjQyOGZkYTYyM2ZlNzg0M2ZjYy5iaW5kUG9wdXAocG9wdXBfNTE5ODZjNGJjMDQ1NGU2Nzk3MmNiODQ1MTc5MGIyN2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzAxYWE3YzdlMjdiNDQyMWE2Yjg0N2YwZDc4ODQxNTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ViNWZiOTA4N2Q1ZjQ0OGI5MGQ1MjIzMDFlZDQ1NDE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U4Y2I2NmZmZGU5YTRjZmFhZmU1YzJmMmJjMzVkOTQ4ID0gJCgnPGRpdiBpZD0iaHRtbF9lOGNiNjZmZmRlOWE0Y2ZhYWZlNWMyZjJiYzM1ZDk0OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogSGFyYm9yZCwgVW5pdmVyc2l0eSBvZiBUb3JvbnRvIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWI1ZmI5MDg3ZDVmNDQ4YjkwZDUyMjMwMWVkNDU0MTQuc2V0Q29udGVudChodG1sX2U4Y2I2NmZmZGU5YTRjZmFhZmU1YzJmMmJjMzVkOTQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcwMWFhN2M3ZTI3YjQ0MjFhNmI4NDdmMGQ3ODg0MTUzLmJpbmRQb3B1cChwb3B1cF9lYjVmYjkwODdkNWY0NDhiOTBkNTIyMzAxZWQ0NTQxNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MWQzNzk3NjM2MjE0YmNhOTQ1NmMwMGQyYTc1Y2FhYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTQ0OWJkZGQ5NDM4NDk0Zjg4MzlkNjc2MjA0Mjc2MzAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWI5NmVkYzQzYzE5NGMwNTlhMDdjMjc1ZDRlMDkyYWMgPSAkKCc8ZGl2IGlkPSJodG1sXzFiOTZlZGM0M2MxOTRjMDU5YTA3YzI3NWQ0ZTA5MmFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvOiBDaGluYXRvd24sIEdyYW5nZSBQYXJrLCBLZW5zaW5ndG9uIE1hcmtldCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E0NDliZGRkOTQzODQ5NGY4ODM5ZDY3NjIwNDI3NjMwLnNldENvbnRlbnQoaHRtbF8xYjk2ZWRjNDNjMTk0YzA1OWEwN2MyNzVkNGUwOTJhYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MWQzNzk3NjM2MjE0YmNhOTQ1NmMwMGQyYTc1Y2FhYy5iaW5kUG9wdXAocG9wdXBfYTQ0OWJkZGQ5NDM4NDk0Zjg4MzlkNjc2MjA0Mjc2MzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjAyYzNmNmE2NzdiNGU3MGJiZmMzYTNlOGYzMzVlZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzljNWM2ZWMzNzFjNTRhZTRhNjVjZTIwZWE1MDY2YTVhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRjM2ZhYjVkZTAzOTQ2MTViOWUyZDIyZjY1YTI5MzUwID0gJCgnPGRpdiBpZD0iaHRtbF80YzNmYWI1ZGUwMzk0NjE1YjllMmQyMmY2NWEyOTM1MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bzogQ04gVG93ZXIsIEJhdGh1cnN0IFF1YXksIElzbGFuZCBhaXJwb3J0LCBIYXJib3VyZnJvbnQgV2VzdCwgS2luZyBhbmQgU3BhZGluYSwgUmFpbHdheSBMYW5kcywgU291dGggTmlhZ2FyYSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzljNWM2ZWMzNzFjNTRhZTRhNjVjZTIwZWE1MDY2YTVhLnNldENvbnRlbnQoaHRtbF80YzNmYWI1ZGUwMzk0NjE1YjllMmQyMmY2NWEyOTM1MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMDJjM2Y2YTY3N2I0ZTcwYmJmYzNhM2U4ZjMzNWVlYS5iaW5kUG9wdXAocG9wdXBfOWM1YzZlYzM3MWM1NGFlNGE2NWNlMjBlYTUwNjZhNWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWNkZmQ2MDI5MTUxNDg5ZGExNTZjZmZlMDJjNDdjZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NDA4N2ZkOWViM2Q0NmM0OGZkNDQ4YzE3NTBiOGU1NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYzYwM2I3ODcwMGM0NzZhOWZiNTBmNmU5NmFkYTM3MCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmM2MDNiNzg3MDBjNDc2YTlmYjUwZjZlOTZhZGEzNzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IFN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NDA4N2ZkOWViM2Q0NmM0OGZkNDQ4YzE3NTBiOGU1Ni5zZXRDb250ZW50KGh0bWxfYmM2MDNiNzg3MDBjNDc2YTlmYjUwZjZlOTZhZGEzNzApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWNkZmQ2MDI5MTUxNDg5ZGExNTZjZmZlMDJjNDdjZmMuYmluZFBvcHVwKHBvcHVwXzk0MDg3ZmQ5ZWIzZDQ2YzQ4ZmQ0NDhjMTc1MGI4ZTU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlZmRjZWI2ZmRiZDRlMTM4N2Y2ZmM0NDkwYzg5ZGQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZDk1OTYyYWMxYTg0ODUyYTM4YjI5NWNmZDQ0NDc1ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZTJkMmJhNTZkYjU0ZWUzODZlZDJiZDQyY2UxMzExZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNWUyZDJiYTU2ZGI1NGVlMzg2ZWQyYmQ0MmNlMTMxMWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IEZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmQ5NTk2MmFjMWE4NDg1MmEzOGIyOTVjZmQ0NDQ3NWUuc2V0Q29udGVudChodG1sXzVlMmQyYmE1NmRiNTRlZTM4NmVkMmJkNDJjZTEzMTFkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlZmRjZWI2ZmRiZDRlMTM4N2Y2ZmM0NDkwYzg5ZGQ4LmJpbmRQb3B1cChwb3B1cF9iZDk1OTYyYWMxYTg0ODUyYTM4YjI5NWNmZDQ0NDc1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MDY2NjNlMDg5Yjc0ZDM2OTAwNjU2MmM1OWZkNzQxNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYTIzOWY3YzdkYWU0MGY4YjI5MjViNjIwZDU2YmFjYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjVjMmE4NmQ5YTA0ZmMwYmQ1M2U3ZjY1OGIwYTU4NCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzY1YzJhODZkOWEwNGZjMGJkNTNlN2Y2NThiMGE1ODQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG86IENocmlzdGllIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWEyMzlmN2M3ZGFlNDBmOGIyOTI1YjYyMGQ1NmJhY2Euc2V0Q29udGVudChodG1sXzM2NWMyYTg2ZDlhMDRmYzBiZDUzZTdmNjU4YjBhNTg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUwNjY2M2UwODliNzRkMzY5MDA2NTYyYzU5ZmQ3NDE2LmJpbmRQb3B1cChwb3B1cF8xYTIzOWY3YzdkYWU0MGY4YjI5MjViNjIwZDU2YmFjYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOGViOTk2NDUyYWM0MGEwYTg2ZjQ2NjVkNjlmYzBmNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBkODYyYmNmMTUyNzRjZGZhOTM3M2E4MjM4ODMwOWYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IwNDAwNmU4NTAwMzQ4ZmY4Yzc4ZjY3MjBiNmViNWFiID0gJCgnPGRpdiBpZD0iaHRtbF9iMDQwMDZlODUwMDM0OGZmOGM3OGY2NzIwYjZlYjVhYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvOiBEb3ZlcmNvdXJ0IFZpbGxhZ2UsIER1ZmZlcmluIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGQ4NjJiY2YxNTI3NGNkZmE5MzczYTgyMzg4MzA5ZjMuc2V0Q29udGVudChodG1sX2IwNDAwNmU4NTAwMzQ4ZmY4Yzc4ZjY3MjBiNmViNWFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q4ZWI5OTY0NTJhYzQwYTBhODZmNDY2NWQ2OWZjMGY0LmJpbmRQb3B1cChwb3B1cF8wZDg2MmJjZjE1Mjc0Y2RmYTkzNzNhODIzODgzMDlmMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNDA2ODQwYmQ0OGY0MTFhYjI3M2IyOTdlY2E0Y2FkZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYjE0YzM5MzJjYmY0MDY2YjI1NmZjZTAwNmVmNjJhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YTgzZmM5ZmM2OTc0NjI2YWQyMDY0NzkxNmMyYjRjMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNWE4M2ZjOWZjNjk3NDYyNmFkMjA2NDc5MTZjMmI0YzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bzogTGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmIxNGMzOTMyY2JmNDA2NmIyNTZmY2UwMDZlZjYyYTguc2V0Q29udGVudChodG1sXzVhODNmYzlmYzY5NzQ2MjZhZDIwNjQ3OTE2YzJiNGMxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M0MDY4NDBiZDQ4ZjQxMWFiMjczYjI5N2VjYTRjYWRmLmJpbmRQb3B1cChwb3B1cF9mYjE0YzM5MzJjYmY0MDY2YjI1NmZjZTAwNmVmNjJhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OWQ1MzlhNWIzOGI0YmRlYWIwMTc1ODAxZDFhNzI3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjg0NzIsLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI1NjAwZTI1NWQzMzQ2ZmE5MjMyZjg0NzkxMGFiYjcwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ3YjNhMDNiOTU5MjQ5YWJiOTQ2NGM5ZjhlZDUxMDAyID0gJCgnPGRpdiBpZD0iaHRtbF80N2IzYTAzYjk1OTI0OWFiYjk0NjRjOWY4ZWQ1MTAwMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvOiBCcm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI1NjAwZTI1NWQzMzQ2ZmE5MjMyZjg0NzkxMGFiYjcwLnNldENvbnRlbnQoaHRtbF80N2IzYTAzYjk1OTI0OWFiYjk0NjRjOWY4ZWQ1MTAwMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82OWQ1MzlhNWIzOGI0YmRlYWIwMTc1ODAxZDFhNzI3MC5iaW5kUG9wdXAocG9wdXBfMjU2MDBlMjU1ZDMzNDZmYTkyMzJmODQ3OTEwYWJiNzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTI1OWUxZjAxMTgyNGNiNGExMmJmYjUzZGQ2M2NhY2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iOTU0NGE1MzFlMzQ0ZmMxYjczNjRlZmFlNTBlZjE0NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMmUzNzlhMDJhMzI0Njc3YjkwOThlODY4MDViMTE4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNzViZDEzOWE1Mzc0OTUyOGU3ZmRmZTA2ZTVkZDhlNiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTc1YmQxMzlhNTM3NDk1MjhlN2ZkZmUwNmU1ZGQ4ZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bzogSGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGggQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMmUzNzlhMDJhMzI0Njc3YjkwOThlODY4MDViMTE4MS5zZXRDb250ZW50KGh0bWxfMTc1YmQxMzlhNTM3NDk1MjhlN2ZkZmUwNmU1ZGQ4ZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTI1OWUxZjAxMTgyNGNiNGExMmJmYjUzZGQ2M2NhY2IuYmluZFBvcHVwKHBvcHVwX2YyZTM3OWEwMmEzMjQ2NzdiOTA5OGU4NjgwNWIxMTgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y3Y2M3MDlkYTMxYzQyNTM4MzZhZjY3MzM0NWUzZGUzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2I5NTQ0YTUzMWUzNDRmYzFiNzM2NGVmYWU1MGVmMTQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FkOGQ4YWM2YjhlMDRjZmJhOGUwMDFlMzk0MmM3N2QwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJiZDI3OTVlZTczODQzYmE5NDBlYzZiMWQ2NGZiMTM3ID0gJCgnPGRpdiBpZD0iaHRtbF8yYmQyNzk1ZWU3Mzg0M2JhOTQwZWM2YjFkNjRmYjEzNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvOiBQYXJrZGFsZSwgUm9uY2VzdmFsbGVzIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWQ4ZDhhYzZiOGUwNGNmYmE4ZTAwMWUzOTQyYzc3ZDAuc2V0Q29udGVudChodG1sXzJiZDI3OTVlZTczODQzYmE5NDBlYzZiMWQ2NGZiMTM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y3Y2M3MDlkYTMxYzQyNTM4MzZhZjY3MzM0NWUzZGUzLmJpbmRQb3B1cChwb3B1cF9hZDhkOGFjNmI4ZTA0Y2ZiYThlMDAxZTM5NDJjNzdkMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MTQxNTQ2NmJjNTk0MGQyOGI2NDgyMWYxOGJkMzVjZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTU3MDYsLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2VjMDUwY2IyNWI1NGYzYmFjMDI2ODRkMmQ3NjNiNTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTIxOTk0OWZiNDgyNGIwNjgyNmQwZjY5ZDk1NDVlNGYgPSAkKCc8ZGl2IGlkPSJodG1sXzEyMTk5NDlmYjQ4MjRiMDY4MjZkMGY2OWQ5NTQ1ZTRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG86IFJ1bm55bWVkZSwgU3dhbnNlYSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NlYzA1MGNiMjViNTRmM2JhYzAyNjg0ZDJkNzYzYjUxLnNldENvbnRlbnQoaHRtbF8xMjE5OTQ5ZmI0ODI0YjA2ODI2ZDBmNjlkOTU0NWU0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MTQxNTQ2NmJjNTk0MGQyOGI2NDgyMWYxOGJkMzVjZC5iaW5kUG9wdXAocG9wdXBfY2VjMDUwY2IyNWI1NGYzYmFjMDI2ODRkMmQ3NjNiNTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjY5NzNlODgzYjEyNDMxMmE3Mjg5Y2E3NGFiY2MxMTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYjk1NDRhNTMxZTM0NGZjMWI3MzY0ZWZhZTUwZWYxNDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWY5MjZiN2ZlODFlNGVmMGFhYzZjODA5Yzk2YzFjMDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWNkYmM4MjlhOWIxNGRhNmIzOWJkZWIwZjYzNzhjMDIgPSAkKCc8ZGl2IGlkPSJodG1sXzFjZGJjODI5YTliMTRkYTZiMzliZGViMGY2Mzc4YzAyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG86IEJ1c2luZXNzIFJlcGx5IE1haWwgUHJvY2Vzc2luZyBDZW50cmUgOTY5IEVhc3Rlcm4gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZjkyNmI3ZmU4MWU0ZWYwYWFjNmM4MDljOTZjMWMwNi5zZXRDb250ZW50KGh0bWxfMWNkYmM4MjlhOWIxNGRhNmIzOWJkZWIwZjYzNzhjMDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjY5NzNlODgzYjEyNDMxMmE3Mjg5Y2E3NGFiY2MxMTguYmluZFBvcHVwKHBvcHVwX2VmOTI2YjdmZTgxZTRlZjBhYWM2YzgwOWM5NmMxYzA2KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Lets examine the clusters closely and name them if possible

##### Neighborhoods with Cluster Label: 0


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>Neighborhood</td>
      <td>Health Food Store</td>
      <td>Pub</td>
      <td>Trail</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Danforth West, Riverdale</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Italian Restaurant</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches West, India Bazaar</td>
      <td>0</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Brewery</td>
      <td>Steakhouse</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Studio District</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Davisville North</td>
      <td>0</td>
      <td>Gym</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Hotel</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>6</th>
      <td>North Toronto West</td>
      <td>0</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Sporting Goods Shop</td>
      <td>Gym / Fitness Center</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Davisville</td>
      <td>0</td>
      <td>Sandwich Place</td>
      <td>Gym</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>0</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Liquor Store</td>
      <td>Sports Bar</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cabbagetown, St. James Town</td>
      <td>0</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Italian Restaurant</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Church and Wellesley</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Restaurant</td>
      <td>Gay Bar</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Harbourfront</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ryerson, Garden District</td>
      <td>0</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Cosmetics Shop</td>
      <td>Café</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>St. James Town</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Berczy Park</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Beer Bar</td>
      <td>Cheese Shop</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Central Bay Street</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Café</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Adelaide, King, Richmond</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>24</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>BBQ Joint</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Harbord, University of Toronto</td>
      <td>0</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>0</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Vietnamese Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>0</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Plane</td>
      <td>Boutique</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Hotel</td>
    </tr>
    <tr>
      <th>29</th>
      <td>First Canadian Place, Underground city</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Christie</td>
      <td>0</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Nightclub</td>
      <td>Convenience Store</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Dovercourt Village, Dufferin</td>
      <td>0</td>
      <td>Pharmacy</td>
      <td>Supermarket</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Little Portugal, Trinity</td>
      <td>0</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Vietnamese Restaurant</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Breakfast Spot</td>
      <td>Grocery Store</td>
      <td>Intersection</td>
    </tr>
    <tr>
      <th>34</th>
      <td>High Park, The Junction South</td>
      <td>0</td>
      <td>Discount Store</td>
      <td>Bar</td>
      <td>Mexican Restaurant</td>
      <td>Café</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0</td>
      <td>Gift Shop</td>
      <td>Coffee Shop</td>
      <td>Bookstore</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Runnymede, Swansea</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>0</td>
      <td>Light Rail Station</td>
      <td>Pizza Place</td>
      <td>Auto Workshop</td>
      <td>Comic Shop</td>
      <td>Recording Studio</td>
    </tr>
  </tbody>
</table>
</div>



##### Neighborhoods with Cluster Label: 1


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Moore Park, Summerhill East</td>
      <td>1</td>
      <td>Trail</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Tennis Court</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rosedale</td>
      <td>1</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Yoga Studio</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



##### Neighborhoods with Cluster Label: 2


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>Roselawn</td>
      <td>2</td>
      <td>Garden</td>
      <td>Home Service</td>
      <td>Dim Sum Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



##### Neighborhoods with Cluster Label: 3


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Forest Hill North, Forest Hill West</td>
      <td>3</td>
      <td>Park</td>
      <td>Trail</td>
      <td>Jewelry Store</td>
      <td>Sushi Restaurant</td>
      <td>Yoga Studio</td>
    </tr>
  </tbody>
</table>
</div>



##### Neighborhoods with Cluster Label: 4


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Lawrence Park</td>
      <td>4</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Bus Line</td>
      <td>Yoga Studio</td>
      <td>Diner</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations about the clusters

Neighborhoods in Cluster-0 has the most common venues related to food. Hence the cluster can be named as the Eatery cluster. 
Neighborhoods in other clusters are not sufficent enough to give a proper name to those clusters 

#  ------End of Assignment-----
