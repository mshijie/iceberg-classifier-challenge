jupyter nbconvert simple_cnn.ipynb --to python
floyd run --gpu --env keras --data iceberg:/data "python simple_cnn.py" 