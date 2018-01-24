jupyter nbconvert convert_data.ipynb --to python
floyd run --gpu --env keras --data iceberg:/data "python convert_data.py" 
