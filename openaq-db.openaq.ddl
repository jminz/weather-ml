CREATE EXTERNAL TABLE `openaq`(
  `location_id` int, 
  `sensors_id` int, 
  `location` string, 
  `datetime` string, 
  `lat` float, 
  `lon` float, 
  `parameter` string, 
  `units` string, 
  `value` float)
PARTITIONED BY ( 
  `locationid` string, 
  `year` string, 
  `month` string)
ROW FORMAT DELIMITED 
  FIELDS TERMINATED BY ',' 
  LINES TERMINATED BY '\n' 
WITH SERDEPROPERTIES ( 
  'escape.delim'='\\') 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3://openaq-data-archive/records/csv.gz'
TBLPROPERTIES (
  'skip.header.line.count'='1', 
  'transient_lastDdlTime'='1743874246')
