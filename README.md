# Introduction
Testing the effectiveness of bonus bets used in online betting agencies using traditional Data Science and Statistical techiniques is a problem I've found extremely interesting to think about since my recent interview for a Data Scientist role. My initial thoughts to tackle this problem was to calculate the distribution of key customer metrics such as the mean bet size and weekly betting frequency both before and after a bonus bet was used to see if there was a measureable difference. To explore this problem further, I generated some simplified synthetic customer betting behaviour to test out some solutions to this problem. 

# Generating Synthetic Data
The first simplification was to only generate data as a weekly time series for only 2020. The program was written so that this could be easily changed in future if required, however it was necessary to start out simple and then expand from there.  

## Total Amount Wagered
To generate semi-realistic synthetic data, some key assumptions were made. The first was that the distribution of the total yearly amount waged for all customers **did not** follow a normal distribution, had a support of <img src="https://render.githubusercontent.com/render/math?math=x\in\(0,\infty)"> and was a right-skewed distribution. The Gamma distribution under specific parameters meets these key requirements:

![Gamma Distribution for various k and theta](/Plots/GammaDist.png)

The default parameters chosen were <img src="https://render.githubusercontent.com/render/math?math=k=5, \theta=150"> which is the blue line in the figure above. The Customer class in the `GenerateCustomers.py` draws from the Gamma distribution to determine the total yearly spend first before generating a betting history. 

## Bet Distribution
The frequency distribution of bets placed over time is not a uniform distribution due to large sporting events like the AFL and NRL grand finals, and the spring racing carnival. To get a simplified 
