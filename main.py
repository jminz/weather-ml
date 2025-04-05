import glob
import json
import os
import shutil
import sys
from datetime import datetime
from io import StringIO
from types import SimpleNamespace

import boto3
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from autogluon.tabular import TabularPredictor
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import clear_output
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
import os

class AQParam:
    def __init__(self, id, name, unit, unhealthyThresholdDefault, desc):
        self.id = id
        self.name = name
        self.unit = unit
        self.unhealthyThresholdDefault = unhealthyThresholdDefault
        self.desc = desc

    def isValid(self):
        if (
            self is not None
            and self.id > 0
            and self.unhealthyThresholdDefault > 0.0
            and len(self.name) > 0
            and len(self.unit) > 0
            and len(self.desc) > 0
        ):
            return True
        else:
            return False

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)


# class AQScenario => Defines an ML scenario including a Location w/ NOAA Weather Station ID
#                     and the target OpenAQ Param.
# Note: OpenAQ data mostly begins sometime in 2016, so using that as a default yearStart value.
class AQScenario:
    def __init__(
        self,
        location=None,
        noaaStationID=None,
        aqParamTarget=None,
        unhealthyThreshold=None,
        yearStart=2016,
        yearEnd=datetime.now().year,
        aqRadiusMiles=10,
        featureColumnsToDrop=None,
    ):
        self.location = location
        self.name = location + "_" + aqParamTarget.name
        self.noaaStationID = noaaStationID
        self.noaaStationLat = 0.0
        self.noaaStationLng = 0.0
        self.openAqLocIDs = []

        self.aqParamTarget = aqParamTarget

        if unhealthyThreshold and unhealthyThreshold > 0.0:
            self.unhealthyThreshold = unhealthyThreshold
        else:
            self.unhealthyThreshold = self.aqParamTarget.unhealthyThresholdDefault

        self.yearStart = yearStart
        self.yearEnd = yearEnd
        self.aqRadiusMiles = aqRadiusMiles
        self.aqRadiusMeters = (
            aqRadiusMiles * 1610
        )  # Rough integer approximation is fine here.

        self.modelFolder = "AutogluonModels"

    def getSummary(self):
        return f"Scenario: {self.name} => {self.aqParamTarget.desc} ({self.aqParamTarget.name}) with UnhealthyThreshold > {self.unhealthyThreshold} {self.aqParamTarget.unit}"

    def getModelPath(self):
        return f"{self.modelFolder}/aq_{self.name}_{self.yearStart}-{self.yearEnd}/"

    def updateNoaaStationLatLng(self, noaagsod_df_row):
        # Use a NOAA row to set Lat+Lng values used for the OpenAQ API requests...
        if (
            noaagsod_df_row is not None
            and noaagsod_df_row["LATITUDE"]
            and noaagsod_df_row["LONGITUDE"]
        ):
            self.noaaStationLat = noaagsod_df_row["LATITUDE"]
            self.noaaStationLng = noaagsod_df_row["LONGITUDE"]
            print(
                f"NOAA Station Lat,Lng Updated for Scenario: {self.name} => {self.noaaStationLat},{self.noaaStationLng}"
            )
        else:
            print("NOAA Station Lat,Lng COULD NOT BE UPDATED.")

    def isValid(self):
        if (
            self is not None
            and self.aqParamTarget is not None
            and self.yearStart > 0
            and self.yearEnd > 0
            and self.yearEnd >= self.yearStart
            and self.aqRadiusMiles > 0
            and self.aqRadiusMeters > 0
            and self.unhealthyThreshold > 0.0
            and len(self.name) > 0
            and len(self.noaaStationID) > 0
        ):
            return True
        else:
            return False

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)


# class AQbyWeatherApp => Main app class with settings, AQParams, AQScenarios, and data access methods...
class AQbyWeatherApp:
    def __init__(
        self, mlTargetLabel="isUnhealthy", mlEvalMetric="accuracy", mlTimeLimitSecs=None
    ):
        self.mlTargetLabel = mlTargetLabel
        self.mlEvalMetric = mlEvalMetric
        self.mlTimeLimitSecs = mlTimeLimitSecs
        self.mlIgnoreColumns = [
            "DATE",
            "NAME",
            "LATITUDE",
            "LONGITUDE",
            "value"
        ]

        self.defaultColumnsNOAA = [
            "DATE",
            "NAME",
            "LATITUDE",
            "LONGITUDE",
            "DEWP",
            "WDSP",
            "MAX",
            "MIN",
            "PRCP",
            "MONTH",
            "VISIB",
            "SLP",
            "MXSPD"
        ]  # Default relevant NOAA columns
        self.defaultColumnsOpenAQ = [
            "day",
            "parameter",
            "unit",
            "average",
        ]  # Default relevant OpenAQ columns

        self.aqParams = {}  # A list to save AQParam objects
        self.aqScenarios = {}  # A list to save AQScenario objects

        self.selectedScenario = None

    def addAQParam(self, aqParam):
        if aqParam and aqParam.isValid():
            self.aqParams[aqParam.name] = aqParam
            return True
        else:
            return False

    def addAQScenario(self, aqScenario):
        if aqScenario and aqScenario.isValid():
            self.aqScenarios[aqScenario.name] = aqScenario
            if self.selectedScenario is None:
                self.selectedScenario = self.aqScenarios[
                    next(iter(self.aqScenarios))
                ]  # Default selectedScenario to 1st item.
            return True
        else:
            return False

    def getFilenameNOAA(self):
        if self and self.selectedScenario and self.selectedScenario.isValid():
            return f"dataNOAA_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}_{self.selectedScenario.noaaStationID}.csv"
        else:
            return ""

    def getFilenameOpenAQ(self):
        if (
            self
            and self.selectedScenario
            and self.selectedScenario.isValid()
            and len(self.selectedScenario.openAqLocIDs) > 0
        ):
            idString = ""
            for i in range(0, len(self.selectedScenario.openAqLocIDs)):
                idString = idString + str(self.selectedScenario.openAqLocIDs[i]) + "-"
            idString = idString[:-1]
            return f"dataOpenAQ_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}_{idString}.csv"
        else:
            return ""

    def getFilenameOther(self, prefix):
        if self and self.selectedScenario and self.selectedScenario.isValid():
            return f"{prefix}_{self.selectedScenario.name}_{self.selectedScenario.yearStart}-{self.selectedScenario.yearEnd}.csv"

    def getNoaaDataFrame(self):
        # ASDI Dataset Name: NOAA GSOD
        # ASDI Dataset URL : https://registry.opendata.aws/noaa-gsod/
        # NOAA GSOD README : https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt
        # NOAA GSOD data in S3 is organized by year and Station ID values, so this is straight-forward
        # Example S3 path format => s3://noaa-gsod-pds/{yyyy}/{stationid}.csv
        # Let's start with a new DataFrame and load it from a local CSV or the NOAA data source...
        noaagsod_df = pd.DataFrame()
        filenameNOAA = self.getFilenameNOAA()

        if os.path.exists(filenameNOAA):
            # Use local data file already accessed + prepared...
            print("Loading NOAA GSOD data from local file: ", filenameNOAA)
            noaagsod_df = pd.read_csv(filenameNOAA)
        else:
            # Access + prepare data and save to a local data file...
            noaagsod_bucket = "noaa-gsod-pds"
            print(
                f"Accessing and preparing data from ASDI-hosted NOAA GSOD dataset in Amazon S3 (bucket: {noaagsod_bucket})..."
            )
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            for year in range(
                self.selectedScenario.yearStart, self.selectedScenario.yearEnd + 1
            ):
                key = f"{year}/{self.selectedScenario.noaaStationID}.csv"  # Compute the key to get
                csv_obj = s3.get_object(
                    Bucket=noaagsod_bucket, Key=key
                )  # Get the S3 object
                csv_string = (
                    csv_obj["Body"].read().decode("utf-8")
                )  # Read object contents to a string
                noaagsod_df = pd.concat(
                    [noaagsod_df, pd.read_csv(StringIO(csv_string))], ignore_index=True
                )  # Use the string to build the DataFrame

            # Perform some Feature Engineering to append potentially useful columns to our dataset... (TODO: Add more to optimize...)
            # It may be true that Month affects air quality (ie: seasonal considerations; tends to have correlation for certain areas)
            noaagsod_df["MONTH"] = pd.to_datetime(noaagsod_df["DATE"]).dt.month

            # Trim down to the desired key columns... (do this last in case engineered columns are to be removed)
            noaagsod_df = noaagsod_df[self.defaultColumnsNOAA]

        return noaagsod_df

    def getOpenAqDataFrame(self):
        # Let's start with a new DataFrame and load it from a local CSV or the NOAA data source...
        aq_df = pd.DataFrame()
        aq_reqUrlBase = "https://api.openaq.org/v3"  # OpenAQ ASDI API Endpoint URL Base (ie: add /locations OR /averages)
        aq_headers = {'X-API-Key': os.getenv('OPENAQ_API_KEY')}

        if (
            self.selectedScenario.noaaStationLat == 0.0
            or self.selectedScenario.noaaStationLng == 0.0
        ):
            print("NOAA Station Lat/Lng NOT DEFINED. CANNOT PROCEED")
            return aq_df

        if len(self.selectedScenario.openAqLocIDs) == 0:
            # We must start by querying nearby OpenAQ Locations for their IDs...
            print("Accessing ASDI-hosted OpenAQ Locations (HTTPS API)...")
            aq_reqParams = {
                "limit": 10,
                "page": 1,
                "offset": 0,
                "sort": "desc",
                "order_by": "id",
                "parameter": self.selectedScenario.aqParamTarget.name,
                "coordinates": f"{self.selectedScenario.noaaStationLat},{self.selectedScenario.noaaStationLng}",
                "radius": self.selectedScenario.aqRadiusMeters,
                "isMobile": "false",
                "sensorType": "reference grade",
                "dumpRaw": "false",
            }
            aq_resp = requests.get(aq_reqUrlBase + "/locations", aq_reqParams, headers=aq_headers)
            aq_data = aq_resp.json()
            print("OpenAQ Locations Response: ", aq_data)
            if aq_data["results"] and len(aq_data["results"]) >= 1:

                for i in range(0, len(aq_data["results"])):
                    self.selectedScenario.openAqLocIDs.append(
                        aq_data["results"][i]["id"]
                    )
                print(
                    f"OpenAQ Location IDs within {self.selectedScenario.aqRadiusMiles} miles ({self.selectedScenario.aqRadiusMeters}m) "
                    + f"of NOAA Station {self.selectedScenario.noaaStationID} at "
                    + f"{self.selectedScenario.noaaStationLat},{self.selectedScenario.noaaStationLng}: {self.selectedScenario.openAqLocIDs}"
                )
            else:
                print(
                    f"NO OpenAQ Location IDs found within {self.selectedScenario.aqRadiusMiles} miles ({self.selectedScenario.aqRadiusMeters}m) "
                    + f"of NOAA Station {self.selectedScenario.noaaStationID}. CANNOT PROCEED."
                )

        if len(self.selectedScenario.openAqLocIDs) >= 1:
            filenameOpenAQ = self.getFilenameOpenAQ()
            if os.path.exists(filenameOpenAQ):
                # Use local data file already accessed + prepared...
                print("Loading OpenAQ data from local file: ", filenameOpenAQ)
                aq_df = pd.read_csv(filenameOpenAQ)
            else:
                # Extract data from Athena
                print("Accessing OpenAQ data via Athena")
                print("Please wait.... this may take several minutes")
                # convert the value of locations to string values and create a tuple
                locationids = tuple(str(x) for x in self.selectedScenario.openAqLocIDs)

                # Use Boto3 to get AccountId and save it in a variable
                account_id = boto3.client('sts').get_caller_identity().get('Account')

                # Connect to Athena
                cursor = connect(s3_staging_dir=f's3://aws-athena-query-results-{account_id}-us-east-1-0c4tmtsy',
                                 region_name='us-east-1').cursor()

                # Execute the Athena Query
                cursor.execute(
                    f"""SELECT datetime,
                        parameter,
                        units,
                        value
                    FROM "AwsDataCatalog"."openaq-db"."openaq"

                    WHERE locationid in %(locationid)s
                        AND parameter = %(parameter)s;"""
                    ,{
                        "locationid": locationids,
                        "parameter": self.selectedScenario.aqParamTarget.name
                        }
                    )
                # get the results of the query in a dataframe with the column headings.
                df = pd.DataFrame(cursor.fetchall())
                df.columns = [desc[0] for desc in cursor.description]

                # convert the datetime column into the datetime datatype with utc=true
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

                # get only the date from the datetime column
                df['date'] = pd.to_datetime(df['datetime']).dt.date

                # convert the value column to a float
                df['value'] = df['value'].astype(float)

                # Group the data by the date and calculate the average (mean) for value
                aq_df = df.groupby('date')['value'].mean().reset_index()

                # Perform some Label Engineering to add our binary classification label => {0=OKAY, 1=UNHEALTHY}
                aq_df[self.mlTargetLabel] = np.where(aq_df["value"] <= self.selectedScenario.unhealthyThreshold, 0, 1)
        return aq_df

    def getMergedDataFrame(self, noaagsod_df, aq_df):
        if len(noaagsod_df) > 0 and len(aq_df) > 0:
            noaagsod_df["DATE"] = pd.to_datetime(noaagsod_df["DATE"])
            aq_df["date"] = pd.to_datetime(aq_df["date"])
            merged_df = pd.merge(
                noaagsod_df, aq_df, how="inner", left_on="DATE", right_on="date"
            )
            merged_df = merged_df.drop(columns=self.mlIgnoreColumns)
            return merged_df
        else:
            return pd.DataFrame()

    def getConfusionMatrixData(self, cm):
        cmData = SimpleNamespace()
        cmData.TN = cm[0][0]
        cmData.TP = cm[1][1]
        cmData.FN = cm[1][0]
        cmData.FP = cm[0][1]

        cmData.TN_Rate = cmData.TN / (cmData.TN + cmData.FP)
        cmData.TP_Rate = cmData.TP / (cmData.TP + cmData.FN)
        cmData.FN_Rate = cmData.FN / (cmData.FN + cmData.TP)
        cmData.FP_Rate = cmData.FP / (cmData.FP + cmData.TN)

        cmData.TN_Output = f"True Negatives  (TN): {cmData.TN} of {cmData.TN+cmData.FP} => {round(cmData.TN_Rate * 100, 2)}%"
        cmData.TP_Output = f"True Positives  (TP): {cmData.TP} of {cmData.TP+cmData.FN} => {round(cmData.TP_Rate * 100, 2)}%"
        cmData.FN_Output = f"False Negatives (FN): {cmData.FN} of {cmData.FN+cmData.TP} => {round(cmData.FN_Rate * 100, 2)}%"
        cmData.FP_Output = f"False Positives (FP): {cmData.FP} of {cmData.FP+cmData.TN} => {round(cmData.FP_Rate * 100, 2)}%"

        return cmData




# Review the pre-defined AQParams and AQScenarios in this cell.
# AQParams are added with default thresholds, which can be overridden on a per-AQScenario basis.
# These AQParams are based on the OpenAQ /parameters API call where isCore=true (https://api.openaq.org/v2/parameters).
# Default thresholds where provided using data from EPA.gov (https://www.epa.gov/criteria-air-pollutants/naaqs-table).
# Confirm and adjust params or thresholds as needed for your needs... Not for scientific or health purposes.

# Instantiate main App class with explicit mlTargetLabel and mlEvalMetric provided...
AQbyWeather = AQbyWeatherApp(mlTargetLabel="isUnhealthy", mlEvalMetric="accuracy")

# Define and add new AQParams...
AQbyWeather.addAQParam(
    AQParam(1, "pm10", "µg/m³", 150.0, "Particulate Matter < 10 micrometers")
)
AQbyWeather.addAQParam(
    AQParam(2, "pm25", "µg/m³", 12.0, "Particulate Matter < 2.5 micrometers")
)
AQbyWeather.addAQParam(AQParam(7, "no2", "ppm", 100.0, "Nitrogen Dioxide"))
AQbyWeather.addAQParam(AQParam(8, "co", "ppm", 9.0, "Carbon Monoxide"))
AQbyWeather.addAQParam(AQParam(9, "so2", "ppm", 75.0, "Sulfur Dioxide"))
AQbyWeather.addAQParam(AQParam(10, "o3", "ppm", 0.070, "Ground Level Ozone"))

# Define available AQ Scenarios for certain locations with their associated NOAA GSOD StationID values...
# NOAA GSOD Station Search: https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day
# NOTE: For Ozone Scenarios, we're generally using 0.035 ppm to override the default threshold.
AQbyWeather.addAQScenario(
    AQScenario("bakersfield", "72384023155", AQbyWeather.aqParams["pm25"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("bakersfield", "72384023155", AQbyWeather.aqParams["pm10"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("bakersfield", "72384023155", AQbyWeather.aqParams["o3"], 0.035)
)
AQbyWeather.addAQScenario(
    AQScenario("fresno", "72389093193", AQbyWeather.aqParams["pm25"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("fresno", "72389093193", AQbyWeather.aqParams["o3"], 0.035)
)
AQbyWeather.addAQScenario(
    AQScenario("visalia", "72389693144", AQbyWeather.aqParams["pm25"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("visalia", "72389693144", AQbyWeather.aqParams["o3"], 0.035)
)
AQbyWeather.addAQScenario(
    AQScenario("los-angeles", "72287493134", AQbyWeather.aqParams["pm25"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("los-angeles", "72287493134", AQbyWeather.aqParams["o3"], 0.035)
)
AQbyWeather.addAQScenario(
    AQScenario("phoenix", "72278023183", AQbyWeather.aqParams["pm25"], None)
)
AQbyWeather.addAQScenario(
    AQScenario("phoenix", "72278023183", AQbyWeather.aqParams["o3"], 0.035)
)

print(f"AQbyWeather.aqParams: {str(len(AQbyWeather.aqParams))}")
print(
    f"AQbyWeather.aqScenarios: {str(len(AQbyWeather.aqScenarios))} (Default Selected: {AQbyWeather.selectedScenario.name})"
)


# Select a Scenario via DROP DOWN LIST to use throughout the Notebook. This will drive the ML process...
# A default "value" is set to avoid issues. Change this default to run the Notebook from start-to-finish for that Scenario.
print("*** CHOOSE YOUR OWN SCENARIO HERE ***")
print("Please select a scenario via the following drop-down-list...")
print("(NOTE: If you change scenario, you must re-run remaining cells to see changes.)")
ddl = widgets.Dropdown(
    options=AQbyWeather.aqScenarios.keys(),
    value=AQbyWeather.aqScenarios["bakersfield_pm25"].name,
)  # <-- DEFAULT / FULL-RUN VALUE
ddl

if ddl.value:
    AQbyWeather.selectedScenario = AQbyWeather.aqScenarios[ddl.value]
    print(AQbyWeather.selectedScenario.getSummary())
    print(AQbyWeather.selectedScenario.toJSON())
else:
    print("Please select a scenario via the above drop-down-list.")


# GET NOAA GSOD WEATHER DATA...
print(AQbyWeather.selectedScenario.getSummary())
noaagsod_df = AQbyWeather.getNoaaDataFrame()
noaagsod_df = noaagsod_df[~noaagsod_df.eq(9999.9).any(axis=1)]

if len(noaagsod_df) >= 1:
    # Update NOAA Station Lat/Lng...
    AQbyWeather.selectedScenario.updateNoaaStationLatLng(noaagsod_df.iloc[0])

    # Save DataFrame to CSV...
    noaagsod_df.to_csv(AQbyWeather.getFilenameNOAA(), index=False)

    # Output DataFrame properties...
    print("noaagsod_df.shape =", noaagsod_df.shape)
    # display(noaagsod_df)
else:
    print("No data is available for this location.")

# GET OPENAQ AIR QUALITY DAILY AVERAGES DATA...
print(AQbyWeather.selectedScenario.getSummary())
aq_df = (
    AQbyWeather.getOpenAqDataFrame()
)  # Gets nearby Location IDs THEN gets associated daily averages.

if len(aq_df) > 0:
    # Output DataFrame properties...
    print("aq_df.shape =", aq_df.shape)
    # display(aq_df)
    aq_df.to_csv(AQbyWeather.getFilenameOpenAQ(), index=False)
else:
    print("No data is available for this location.")

# Merge the NOAA GSOD weather data with our OpenAQ data by DATE...
# Perform another column drop to remove columns we don't want as features/inputs.
# This column removal will NOT be necessary once we can use Autogluon ignore_columns param (TBD).
print(AQbyWeather.selectedScenario.getSummary())
merged_df = AQbyWeather.getMergedDataFrame(noaagsod_df, aq_df)

if len(merged_df) > 0:
    # Output DataFrame properties...
    print("merged_df.shape =", merged_df.shape)
    # display(merged_df)
    merged_df.groupby([AQbyWeather.mlTargetLabel]).size().plot(kind="bar")
    merged_df.to_csv(AQbyWeather.getFilenameOther("dataMERGED"), index=False)

# Visualize correlations in our merged dataframe...
print(AQbyWeather.selectedScenario.getSummary())
correlations = merged_df.corr()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1,cmap = "coolwarm")
fig.colorbar(cax)
ticks = np.arange(0, len(correlations.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(correlations.columns)
ax.set_yticklabels(correlations.columns)
# plt.show()

# Split out train_df + validate_df data...
print(AQbyWeather.selectedScenario.getSummary())
train_df, validate_df = train_test_split(merged_df, test_size=0.2, random_state=1)
print('Number of training samples:', len(train_df))
print('Number of validation samples:', len(validate_df))

# Create the test_df data and remove the target label column...
print(AQbyWeather.selectedScenario.getSummary())
test_df = validate_df.drop([AQbyWeather.mlTargetLabel], axis=1)
#display(test_df)

# This section of code may take 5-10 minutes to run. Output may have red background.
# Use AutoGluon TabularPredictor to fit a model for our training data...
#display(AQbyWeather.selectedScenario.getSummary()) #Using display for consistent/sequential output order.
predictor = TabularPredictor(label=AQbyWeather.mlTargetLabel,
                             eval_metric=AQbyWeather.mlEvalMetric,
                             path=AQbyWeather.selectedScenario.getModelPath())
predictor.fit(train_data=train_df, time_limit=AQbyWeather.mlTimeLimitSecs, verbosity=2, presets='best_quality')

# Get dataframes for feature importance + model leaderboard AND get+display model evaluation...
#display(AQbyWeather.selectedScenario.getSummary()) #Using display for consistent/sequential output order.
featureimp_df   = predictor.feature_importance(validate_df)
leaderboard_df  = predictor.leaderboard(validate_df, silent=True)
modelEvaluation = predictor.evaluate(validate_df, auxiliary_metrics=True)

# View Autogluon Individual Model Leaderboard...
print(AQbyWeather.selectedScenario.getSummary())
# display(leaderboard_df)

# View and Plot Feature Importance... (this various from Scenario to Scenario)
print(AQbyWeather.selectedScenario.getSummary())
#display(featureimp_df)
featureimp_df[["importance", "stddev"]].plot(kind="bar", figsize=(12, 4), xlabel="feature")