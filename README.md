# Microgrids

This is the first modeling I did by myself so I wanted to document it using GitHub.

## Goals

What I had in mind was: given inputs, the model will predict the amount of energy consumed within the area of Tetouan. Knowing the amount of energy that will be consumed,
it might be interesting to see if solar energy would be able to fulfill the necessity of electrical power from a grid. Given that, I want to see what varieties can 
happen (the energy is not enough, the energy is more than enough and what to do with that...) and how can a Microgrid participate in that story.

Indeed I was able to predict both energy consumed and solar radiation (that is used to be transformed on electrical power) but I had to use different Datasets, which make things
a little hard to interpret. Would be great if there was a Dataset with both energy consumption and solar radiation as it would make more sense, though the logic used in my models
would be almost perfectly the same.

## Conclusion

The predictions were made but the Microgrid concept was not really applied since it would be the next step after finding the amount of energy produced by the solar panels. It would
take me more time to study how microgrids work (numerically and analytically) so that calculations can be made and the project is actually useful. 

It was an interesting experience as I was able to make many correlations, figure things out by myself, and actually apply engineering. Also, Microgrids seem like a very interesting
subject to study. I didn't go too far on it because it was seamless I wouldn't understand anything by the amount of knowledge I have from college, so all I did was try to understand
how things work: grids, on-grids, off-grids, smart-grids, and "create" relations with that and the knowledge I have on machine learning and data analysis to actually give it an
appliance. 

## Notes

There is a lot to improve on the models: trying other models, loss function, giving them new features and correlations, and applying some statistical concepts (errors, variances...) but it wasn't my goal  to make
something perfect and applicable. What I did was mostly for fun as an Electrical Engineer student from the 3rd semester that knows nothing about the "electrical part" itself. 
