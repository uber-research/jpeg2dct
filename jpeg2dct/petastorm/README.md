# integration with petastorm

allows to use the jpeg2dct codec with petastorm distributed training data storage (https://github.com/uber/petastorm).
Petastorm and underlying dependencies such as pyspark and cv2 needs to be installed.


## Usage
#### Create a petastorm store
```python
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import ScalarCodec
from pyspark.sql.types import IntegerType
from jpeg2dct.petastorm.codecs import Jpeg2DCTNumpyCodec
from petastorm.etl.dataset_metadata import materialize_dataset
import numpy as np

HelloWorldSchema = Unischema('HelloWorldSchema', [
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('image1', np.uint8, (128, 256, 3), Jpeg2DCTNumpyCodec(), False),
])

def row_generator(x):
    """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
    return {'id': x,
            'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3))}

output_url='file:///tmp/jpeg2dct'
rows_count = 10
rowgroup_size_mb = 256

spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
sc = spark.sparkContext

# Wrap dataset materialization portion. Will take care of setting up spark environment variables as
# well as save petastorm specific metadata
with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):
    rows_rdd = sc.parallelize(range(rows_count)).map(row_generator).map(lambda x: dict_to_spark_row(HelloWorldSchema, x))
    spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()).coalesce(10).write.mode('overwrite').parquet(output_url)
```

#### Read a petastorm store
```python
from petastorm.reader import Reader
with Reader(output_url = 'file:///tmp/jpeg2dct') as r:
    for row in r:
        print row
```