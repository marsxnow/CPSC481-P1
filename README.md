# CPSC481-P1

# adding data

To add data you gotta add images into  
`models/datasets/archive/myImg/`
images you add should be jpg or jpeg
when labling the data start with the:

- age,
- followed by gender (0 male, 1 female),
- followed by another 0 and
- followed by an ID number

## example:

for a 25 year old female we'd lable the image something like this
`25_1_0_123413.jpg`

for a 20 year old male we'd lable the image something like this
`20_0_0_1232334.jpg`

# training the model

Once you are satisfied witht the amount of images you put in you can train the model
To train the model make sure you are in
`models/` and run `python3 ageTrain.py`
It'll take a min to load but something should show in the terminal

# to run the app

1. Activate virutal env: `source venv/bin/activate`
2. `python3 app.py`
