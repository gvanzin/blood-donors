import pandas as pd
import graphlab as gl



#%matplotlib inline
#graphlab.canvas.set_target('ipynb')

def load_data():
	'''input:  csv exported from sql
	   output:  cleaned pandas DataFrame'''
	data = pd.read_csv('/Users/garyvanzin/Documents/Terumo_files/tps/TPS_View_10.csv', parse_dates=['Boot_Datetime','Timestamp'])
	#drop columns 
	data.drop(['\xef\xbb\xbfCategory','Country','Attempted_Procedure','Parent_ID','Region_ID','Stamp','Device_ID','Create_Date','Update_Date','Update_User_Id','ConfigTimeFormat','Procedure_Type'],axis=1, inplace=True)
	#the last 7 digits of Raw_Scan_Data are the unique identifier for an individual donor
	data['Donor_id']=data['Raw_Scan_Data'].str[-7:]
	return data


def subset_data():
	'''subset data'''
	data=data[data['Donor_Height']<150]
	data=data[data['Boot_Datetime']<'2015-10-01 00:00:00.000']
	data['start']=data['Raw_Scan_Data'].str[2:]
	data = data[data.start != "-L"]
	data = data[data.Donor_Blood_Type !=0]
	return data

def max_timestamp():
	'''each donation has multiple rows.  The row containing the max value of "timestamp" is used going forward'''
	max_timestamp=data.groupby(['Trima_Summary_ID'], as_index=False,sort=False)['Timestamp'].max()
	data=data.merge(max_timestamp, on=['Trima_Summary_ID','Timestamp'])
	return data

def get_target():
	'''count the number of times each donor has donated.  Add 'target' as count of 1 or more ="1" else "0"'''
	countdf=pd.DataFrame(data.groupby('Donor_id',as_index=False).size(), columns=['count'])
	data=data.join(countdf, on='Donor_id')
	data['target'] = data['count'].apply(lambda x: '1' if x>1 else '0')
	data_balanced=data
	return data_balanced

def date_formatting():
	'''reformat date'''
	data_balanced['Boot_Datetime'] =data_balanced['Boot_Datetime'].map(lambda t: t.strftime('%Y-%m-%d %H:%M:%S'))
	data_balanced['Timestamp'] =data_balanced['Timestamp'].map(lambda t: t.strftime('%Y-%m-%d %H:%M:%S'))
	return data_balanced
	


def bin_donation_count():
	'''bin the number of donations per individual by quantiles'''
	data_balanced['count_group']=pd.qcut(data_balanced['count'], 5, labels=["1","2","3","4","5"])
	data_balanced['count_group']=data_balanced['count_group'].astype(int)
	return data_balanced

def date_engineering():

	data_balanced['day_of_week'] = data_balanced['Boot_Datetime'].dt.dayofweek
	data_balanced['month'] = data_balanced['Boot_Datetime'].dt.month
	data_balanced['hour'] = data_balanced['Boot_Datetime'].dt.hour
	data_balanced['year'] = data_balanced['Boot_Datetime'].dt.year
	data_balanced['week'] = data_balanced['Boot_Datetime'].dt.week

	data_balanced['first_donation_day_of_week'] = data_balanced['Timestamp'].dt.dayofweek
	data_balanced['first_donation_month'] = data_balanced['Timestamp'].dt.month
	data_balanced['first_donation_hour'] = data_balanced['Timestamp'].dt.hour
	data_balanced['first_donation_year'] = data_balanced['Timestamp'].dt.year
	data_balanced['first_donation_week'] = data_balanced['Timestamp'].dt.week
	#remove 2012 data, the few observations are skewed.
	data_balanced=data_balanced[data_balanced['year']!=2012]
	return data_balanced

def products():
	'''get the number of  products produced during the procedure'''

	data_balanced['yield_units']=data_balanced['Platelet_Yield']/3
	data_balanced['yield_units']=data_balanced['yield_units'].astype(int)

	data_balanced['Bag_Volume_units']=data_balanced['Plasma']/200
	data_balanced['Bag_Volume_units']=data_balanced['Bag_Volume_units'].astype(int)

	data_balanced['Bag_Dose_RBC1_units']=data_balanced['RBC']/180
	data_balanced['Bag_Dose_RBC1_units']=data_balanced['Bag_Dose_RBC1_units'].astype(int)

	data_balanced['Products']=data_balanced['yield_units']+data_balanced['Bag_Volume_units']+data_balanced['Bag_Dose_RBC1_units']
	return data_balanced


def convert_to_sframe():
	'''convert pandas dataframe to a graphlab sframe for modeling.  Clean a bit.'''
	sf= gl.SFrame(data_balanced)
	sf['Donor_Gender'] = sf['Donor_Gender'].astype(str)
	sf['day_of_week'] = sf['day_of_week'].astype(str)
	sf['hour'] = sf['hour'].astype(str)
	sf['year'] = sf['year'].astype(str)
	sf['week'] = sf['week'].astype(str)
	sf['month'] = sf['month'].astype(str)
	sf['first_donation_day_of_week'] = sf['first_donation_day_of_week'].astype(str)
	sf['first_donation_hour'] = sf['first_donation_hour'].astype(str)
	sf['first_donation_year'] = sf['first_donation_year'].astype(str)
	sf['first_donation_week'] = sf['first_donation_week'].astype(str)
	sf['first_donation_month'] = sf['first_donation_month'].astype(str)
	sf['BCT_Account_Id'] = sf['BCT_Account_Id'].astype(str)
	sf['Donor_Blood_Type'] = sf['Donor_Blood_Type'].astype(str)
	return sf


def modeling_prep():
	'''remove features not to be used in grid searching'''
	sf.remove_columns(['Raw_Scan_Data', 'Boot_Datetime', 'Account_ID', 'ID', 'Trima_Summary_ID', 'Timestamp', 'count_group'])
	train_set, test_set = sf.random_split(0.8, seed=1)
	return train_set, test_set




if __name__ == '__main__':
	data = load_data()
	print data.head()
	








