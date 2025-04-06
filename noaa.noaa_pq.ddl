CREATE EXTERNAL TABLE `noaa_pq`(
  `station` string, 
  `date` string, 
  `latitude` string, 
  `longitude` string, 
  `elevation` string, 
  `name` string, 
  `temp` string, 
  `temp_attributes` string, 
  `dewp` string, 
  `dewp_attributes` string, 
  `slp` string, 
  `slp_attributes` string, 
  `stp` string, 
  `stp_attributes` string, 
  `visib` string, 
  `visib_attributes` string, 
  `wdsp` string, 
  `wdsp_attributes` string, 
  `mxspd` string, 
  `gust` string, 
  `max` string, 
  `max_attributes` string, 
  `min` string, 
  `min_attributes` string, 
  `prcp` string, 
  `prcp_attributes` string, 
  `sndp` string, 
  `frshtt` string)
PARTITIONED BY ( 
  `year` string)
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
LOCATION
  's3://athena-examples-us-east-1/athenasparksqlblog/noaa_pq'
TBLPROPERTIES (
  'transient_lastDdlTime'='1743870338')
