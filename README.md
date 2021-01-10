# Introduction
Testing the effectiveness of bonus bets used in online betting agencies using traditional Data Science and Statistical techiniques is a problem I've found extremely interesting to think about since my recent interview for a Data Scientist role. My initial thoughts to tackle this problem was to calculate the distribution of key customer metrics such as the mean bet size and weekly betting frequency both before and after a bonus bet was used to see if there was a measureable difference. To explore this problem further, I generated some simplified synthetic customer betting behaviour to test out some solutions to this problem. 

The approach taken was to simpily draw samples from probability distributions for different customer behaviours based on some simplified assumptions and change the underlying distributions after a bonus bet date was decided and then measure the resulting change in the distributions. 

# Generating Synthetic Data
The first simplification was to only generate data as a weekly time series for only 2020. The program was written so that this could be easily changed in future if required, however it was necessary to start out simple and then expand from there.  

## Total Amount Wagered
To generate semi-realistic synthetic data, some key assumptions were made. The first was that the distribution of the total yearly amount waged for all customers **did not** follow a normal distribution, had a support of <img src="https://render.githubusercontent.com/render/math?math=x\in\(0,\infty)"> and was a right-skewed distribution. The [`Gamma distribution`](https://en.wikipedia.org/wiki/Gamma_distribution) under specific parameters meets these key requirements:

![Gamma Distribution for various k and theta](/Plots/GammaDist.png)

The default parameters chosen were <img src="https://render.githubusercontent.com/render/math?math=k=5, \theta=150"> which is the blue line in the figure above. The Customer class in the `GenerateCustomers.py` draws from the Gamma distribution to determine the total yearly spend first before generating a betting history. 

## Bet Frequency Distribution
The frequency distribution of bets placed over time is not a uniform distribution due to large sporting events like the AFL and NRL grand finals, State of Origin, and the spring racing carnival. To get a simplified approximation of this distribution [`Google Trends`](https://trends.google.com/trends/?geo=AU) can be used, which returns the region specific interest in search terms or topics. The weekly distribution for 2020 for the search term 'betting' is shown below

![Google Trends for Betting](/Plots/Google%20Trends.png)

with the large peak in the distribution due to the Melbourne Cup. This approximate distribution was used to generate the betting frequency for every customer. In reality, each customers betting distribution will be highly dependent on their sport preferences.

## Individual Bets
To generate a betting history for each customer, an individual bet size was calculated which was again drawn from a Gamma distribution with shape and shape parameters <img src="https://render.githubusercontent.com/render/math?math=k=10, \theta=2">. The date of the bet was drawn from the frequency distribution discussed above. This process was repeated until the total yearly amount waged was reached which returned the full customers betting history. This is implimented in the `generate_bets` method in the `Customers` class of `GenerateCustomers.py` and the resulting betting history is stored in the `betting_history` property. 

This method was easy to impliment quickly, however in reality one would suspect the betting distribution for each customer could be conditional on their total yearly amount wagered, something that would be interesting to investigate. 

## Adding A Bonus Bet
A random week was selected to apply the bonus bet. This random date could either be the same for every customer or different and is controlled through the `random_bonusbet` parameter in `GenerateCustomers.py`. Once the bonus bet date had been determined the bet frequency and inividual bet size distributions are changed using the `bonusbet_sizemultiplier` and `bonusbet_freqmultiplier` parameters respectively. See the comments in the `Customer` class for more detail if interested. 


# Output
To generate 10,000 customers using the default parameters and store each customer in a dictionary, pythons dictonary comprehension can be used
```python
customers = {i: Customer(customer_id=i) for i in range(1,10001)}
```
now each customer is stored in memory in a dictoary item. This is not the most computationally efficent method but saves the need for a relational database schemea to be designed and implemented, which would be more time consuming and outside the scope of doing something simple and quick. 

The betting history for each customer is stored in the `betting_history` property of the Customer class. To combine them into a single Pandas DataFrame, first create an emtpy Dataframe with the same column names as the `betting_history` and the merge each DataFrame together:

```python
total_customer_betting_df = pd.DataFrame(columns=customers[1].betting_history.columns)
for customer_id, customer in customers.items():
    total_customer_betting_df = pd.concat([total_customer_betting_df, customer.betting_history])              
```
From this we can see the total yearly time-series follows a similar distribution to what 
![Total Series](/Plots/Total%20Betting%20Distribution.png)
