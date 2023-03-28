import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


features = list()

with open(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\features.txt', 'r') as f:
    for line in f:
        features.append(line.split()[1])


train_df = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\train\X_train.txt', delim_whitespace=True, header = None)
train_df.columns = features
train_df['subject_id'] = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\train\subject_train.txt', header=None, squeeze=True)
train_df['activity'] = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\train\y_train.txt', header=None, squeeze=True)
activity = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\train\y_train.txt', header = None, squeeze = True)



test_df = pd.read_csv(r"C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\test\x_test.txt", delim_whitespace = True, header = None)
test_df.columns = features


test_df['subject_id'] = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\test\subject_test.txt', header=None, squeeze=True)

test_df['activity'] = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\test\y_test.txt', header=None, squeeze=True)

activity = pd.read_csv(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\test\y_test.txt', header=None, squeeze=True)

# Load the activity labels
with open(r'C:\Users\Vishal Sahni\Desktop\Docu3c_Internship\Gait_Analysis\UCI HAR Dataset\activity_labels.txt', 'r') as f:
    activity_labels = [line.strip().split()[1] for line in f.readlines()]

df = pd.concat([train_df, test_df], axis=0)

# Replace activity codes with labels
df['activity_label'] = df['activity'].map(lambda x: activity_labels[x-1])


# Create a histogram of the activity labels
activity_counts = df['subject_id'].value_counts()
plt.bar(activity_counts.index, activity_counts.values)
plt.title('Activity Distribution')
plt.xlabel('Subject_id')
plt.ylabel('Number of Instances')
st.pyplot()


# Create a line chart of the accelerometer data for a single activity
activity = st.selectbox('Select an Subject_id', df['subject_id'].unique())
activity_df = df[df['subject_id'] == activity]
plt.plot(activity_df['tBodyAcc-mean()-X'], label='X')
plt.plot(activity_df['tBodyAcc-mean()-Y'], label='Y')
plt.plot(activity_df['tBodyAcc-mean()-Z'], label='Z')
plt.title(f'Accelerometer Data for {activity}')
plt.xlabel('Time')
plt.ylabel('Acceleration (g)')
plt.legend()
st.pyplot()


# Define stationary and moving activities
stationary_activities = ['LAYING', 'SITTING', 'STANDING']
moving_activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']

# Create a FacetGrid plot
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(df, hue='activity_label', size=6, aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()

# Add annotations for stationary and moving activities
plt.annotate("Stationary Activities", xy=(-0.956, 17), xytext=(-0.9, 23), size=20, va='center', ha='left', arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))
plt.annotate("Moving Activities", xy=(0, 3), xytext=(0.2, 9), size=20, va='center', ha='left', arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))

# Set the x and y axis labels
plt.xlabel('tBodyAccMag-mean')
plt.ylabel('Density')

# Show the plot in Streamlit
st.pyplot()





# Create a boxplot of the tBodyAccMagmean for each activity
plt.figure(figsize=(7,7))
sns.boxplot(x='activity_label', y='tBodyAccMag-mean()', data=df, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9, dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)

# Display the plot using Streamlit
st.pyplot()

