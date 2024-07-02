# FlatBuffer notes

## Download the schema

```bash
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema_v3c.fbs
```

## Reading the model to JSON

(`flatc` should have been installed with the `flatbuffers` conda package, which is pulled in by the `tensorflow` package)

```bash
flatc -t --strict-json --defaults-json schema_v3c.fbs -- weights.bin
```

## Converting the schema to Python boilerplate

```bash
flatc --python --python-typing schema_v3c.fbs
```