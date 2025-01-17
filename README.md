# Introduction
Testing the effectiveness of bonus bets used in online betting agencies using traditional Data Science and Statistical techniques is a problem I've found extremely interesting to think about since my recent interview for a Data Scientist role. My initial thoughts to tackle this problem was to calculate the distribution of key customer metrics such as the mean bet size and weekly betting frequency both before and after a bonus bet was used to see if there was a measurable difference. To explore this problem further, I generated some simplified synthetic customer betting behaviour to test out some solutions to this problem. 

The approach taken was to simply draw samples from probability distributions for different customer behaviours based on some simplified assumptions. Then after calculating a random bonus bet date, change these underlying distributions and try and develop a technique to measure the change. 

# Generating Synthetic Data
The first simplification was to only generate data as a weekly time series for only 2020. The program was written so that this could be easily changed in the future if required, however, it was necessary to start out simple and then expand from there.  

## Total Amount Wagered
To generate semi-realistic synthetic data, some key assumptions were made. The first was that the distribution of the total yearly amount waged for all customers **did not** follow a normal distribution, had a support of <img src="https://render.githubusercontent.com/render/math?math=x\in\(0,\infty)"> and was a right-skewed distribution. The [`Gamma distribution`](https://en.wikipedia.org/wiki/Gamma_distribution) under specific parameters meets these key requirements:

![Gamma Distribution for various k and theta](/Plots/GammaDist.png)

The default parameters chosen were <img src="https://render.githubusercontent.com/render/math?math=k=5, \theta=150"> which is the blue line in the figure above. The Customer class in the `GenerateCustomers.py` draws from the Gamma distribution to determine the total yearly spend first before generating a betting history. 

## Bet Frequency Distribution
The frequency distribution of bets placed over time is not a uniform distribution due to large sporting events like the AFL and NRL grand finals, State of Origin, and the spring racing carnival. To get a simplified approximation of this distribution [`Google Trends`](https://trends.google.com/trends/?geo=AU) can be used, which returns the region specific interest in search terms or topics. The weekly distribution for 2020 for the search term 'betting' is shown below

![Google Trends for Betting](/Plots/Google%20Trends.png)

with the large peak in the distribution due to the Melbourne Cup. This approximate distribution was used to generate the betting frequency for every customer. In reality, each customers betting distribution will be highly dependent on their sport preferences.

## Individual Bets
To generate a betting history for each customer, an individual bet size was calculated which was again drawn from a Gamma distribution with shape and shape parameters <img src="https://render.githubusercontent.com/render/math?math=k=10, \theta=2">. The date of the bet was drawn from the frequency distribution discussed above. This process was repeated until the total yearly amount waged was reached which returned the full customers betting history. This is implemented in the `generate_bets` method in the `Customers` class of `GenerateCustomers.py` and the resulting betting history is stored in the `betting_history` property. 

This method was easy to implement quickly, however in reality one would suspect the betting distribution for each customer could be conditional on their total yearly amount wagered, something that would be interesting to investigate. 

## Adding A Bonus Bet
A random week was selected to apply the bonus bet. This random date could either be the same for every customer or different and is controlled through the `random_bonusbet` parameter in `GenerateCustomers.py`. Once the bonus bet date had been determined the bet frequency and individual bet size distributions are changed using the `bonusbet_sizemultiplier` and `bonusbet_freqmultiplier` parameters respectively. See the comments in the `Customer` class for more detail if interested. 


# Output
To generate 10,000 customers using the default parameters and store each customer in a dictionary, pythons dictonary comprehension can be used
```python
customers = {i: Customer(customer_id=i) for i in range(1,10001)}
```
now each customer is stored in memory in a dictionary item. This is not the most computationally efficient method but saves the need for a relational database schema to be designed and implemented, which would be more time consuming and outside the scope of doing something simple and quick. 

## Data Exploration

The betting history for each customer is stored in the `betting_history` property of the Customer class. To combine them into a single Pandas DataFrame, first create an emtpy Dataframe with the same column names as the `betting_history` and the merge each DataFrame together:

```python
total_customer_betting_df = pd.DataFrame(columns=customers[1].betting_history.columns)
for customer_id, customer in customers.items():
    total_customer_betting_df = pd.concat([total_customer_betting_df, customer.betting_history])              
```
From this, the total yearly time-series follows a similar distribution to the assumed betting frequency distribution from Google Trends, as expected - a positive sign there are no bugs in the code.  

![Total Series](/Plots/Total%20Betting%20Distribution.png)

As I used the default parameters, each customer has the same date the bonus bet, which can be easily seen in the figure above as there is a clear change in the distribution. To randomise the date of the bonus bet for each customer, the `random_bonusbet` parameter can be set to True
```python
customers = {i: Customer(customer_id=i,random_bonusbet=True) for i in range(1, 10001)}
```
which returns a much more realistic frequency distribution and one that is hard to detect if there is a change.

![Total Series With Random Bonus Bet](/Plots/Total%20Betting%20Distribution%20Random%20Date.png)

# Detecting Changes in Distributions

To measure a change between two distributions the [`Jensen-Shannon Divergence`](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) can be used as it is a method of measuring the similarity between two probability distributions. It has a bound between 0 and 1, with 0 identical distributions and 1 non-overlapping distributions. The `calculate_jensenshannon` method in the `GenerateCustomers.py` program takes a dictonary of customers as an input and then returns the Jensen-Shannon Divergence of the distributions before and after a bonus bet for both the mean bet size and weekly bet frequency.

The mean bet size is calculated before and after the bonus bet and the distribution is over the total number of customers in the dictonary. It is important to use the mean bet size instead of the total as time scale can be different either side of the bonus bet. For example, if the bonus bet is in March, there will be a difference in the total amount before and after the bonus bet just because there is a longer time period after the March bonus bet date. 

## Applying the Jensen-Shannon Divergence 
The Jensen-Shannon Divergence can be used to measure the difference in two distributions, to test it is sensitive enough to detect a change we can set up a loop to tweak how much we change the bet amount and bet frequency distributions

```python
freq_js = []
amount_js = []
counter = 1
linspace_size = 11
# loop over a size and frequency multiplier
for bonus_bet_size_multiplier in np.linspace(1, 1.5, linspace_size):
    for bonus_bet_freq_multiplier in np.linspace(1, 1.5, linspace_size):
        # quick set up of a loop counter
        print("Loop {} of {}".format(counter, linspace_size**2))
        counter += 1
        # create 1,000 customers with random bonus bet dates with different size and frequency multipliers
        customers = {i: Customer(i,
                                 bonusbet_sizemultiplier=bonus_bet_size_multiplier,
                                 bonusbet_freqmultiplier=bonus_bet_freq_multiplier,
                                 random_bonusbet=True
                                 ) for i in range(1, 1001)}
        # return the Jensen-Shannon divergence
        result = calculate_jensenshannon(customers, True)
        # save the results to a list
        amount_js.append(result[0])
        freq_js.append(result[1])
```

The result of this gives the change in Jensen-Shannon divergence as a function of how much we change the bet size and bet frequency multipliers. Shown below is the Jensen-Shannon divergence for distributions of the mean bet size before and after a bonus bet

![mean bet JS](/Plots/Amount_JS.png)

It clearly shows that as the bet size multiplier is increased, the Jensen-Shannon divergence increases. Changing the frequency distribution has no change on the Jensen-Shannon divergence, as expected. Plotting the Jensen-Shannon divergence for distributions of the bet frequency before and after a bonus bet gives a similar result

![mean bet JS](/Plots/Frequency_JS.png)

Interestingly, there is a difference in the scale between the mean bet size and frequency plots, suggesting it would be easier to detect changes in the bet size after a bonus bet. 

## Problems with the Jensen-Shannon Approach
While the Jensen-Shannon approach gives some ok results for distributions when we explicitly add a change in the underlying distributions, in practice it may not be useful to use as there are some underlying problems. Firstly, the Jensen-Shannon Divergence gives no indication to the direction of the change in distribution, thus if the bonus bet drove away customers, a change would be detected but no extra infromation would be available. Secondly, to apply the Jensen-Shannon Divergence to two distributions and make sense of the output, you need have some knowledge of what to expect if there was no change in the underlying distributions. 

# Further Extensions
Some interesting further extensions to this program to better simulate reality would be:
* Add multiple bonus bets over a period of time 
* Add a response time to the bonus bet so that it only is effective for a period of time
* Change the bet size multiplier to a distribution instead of a constant
* Add proper Bayesian inference to the distributions
