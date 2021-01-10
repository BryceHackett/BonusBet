# import standard data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import jensenshannon
# import Path to read the csv file in an OS independent way
from pathlib import Path
# Use pickle to save temporary results
import pickle


# Create a customer class to store each customers betting history and parameters
class Customer(object):
    """
    class to create a customer to generate synthetic betting behaviour
    Parameters
    ----------
    customer_id: int
    user defined numeric customer id

    bonusbet_sizemultiplier: float -> default=1.2
    size multiplier for the change in betting history after a bonus bet

    bonusbet_freqmultiplier: float -> default=1.2
    frequency multiplier for the change in betting history after a bonus bet

    random_bonusbet: bool -> default=False
    Random bonus bet for each customer if True

    gamma_param: tuple dim 2 -> (5,150)
    Paramters for the gamma distribution to control the betting

    Returns
    ----------
    A Class instance of the customer betting history and parameters stored in the class
    """

    def __init__(self,
                 customer_id,
                 bonusbet_sizemultiplier=1.2,
                 bonusbet_freqmultiplier=1.2,
                 random_bonusbet=False,
                 gamma_param=(5, 150)):
        # Set the Customer ID
        self.customer_id = customer_id
        # Generate the total yearly spend from the gamma distribution
        # This distribution has some nice properties, it's support is [0,\inf) (i.e. non-negative)
        # Under the parameters chosen, it's a highly right-skewed distribution
        # Which I'd assume the customer behaviour would be in reality
        self.total_spend = np.random.gamma(gamma_param[0], gamma_param[1])
        # Set the bonus bet size multiplier
        self.bonusbet_sizemultiplier = bonusbet_sizemultiplier
        # Set the bonus bet frequency multiplier
        self.bonusbet_freqmultiplier = bonusbet_freqmultiplier
        # Read in the racing CSV from google trends as an approximation for the betting distribution over time
        # To simplify things only use the aggregate weekly series
        # TODO: change this so it's not read in for every single customer
        self.racing_df = pd.read_csv(Path().joinpath("Data", "RacingTrendsGoogle.csv"),
                                     index_col="Date", parse_dates=True)
        # If there's a random bonus bet for each customer (i.e. each bonus bet was applied in different weeks)
        # Take a different random sample of the weeks available for each unique customer
        # Else set the seed so every customer gets the same bonus bet date
        if random_bonusbet:
            self.random_bonusbet_date = self.racing_df.sample(1).index[0]
        else:
            self.random_bonusbet_date = self.racing_df.sample(1, random_state=42).index[0]
        # Change the frequency distribution by adding a multiplier to the distribution
        # TODO: Replace these line with a single call, hack way to change the data frequency distribution
        self.racing_df.loc[self.racing_df.index >= self.random_bonusbet_date, 'Adjusted Proportion'] = \
            self.racing_df.loc[self.racing_df.index >= self.random_bonusbet_date, "Proportion"] * \
            self.bonusbet_freqmultiplier
        self.racing_df.loc[self.racing_df.index < self.random_bonusbet_date, 'Adjusted Proportion'] = \
            self.racing_df.loc[self.racing_df.index < self.random_bonusbet_date, "Proportion"]
        # Normalise the probability to sample from
        self.racing_df['Racing Prob'] = self.racing_df["Adjusted Proportion"] / \
                                        self.racing_df["Adjusted Proportion"].sum()
        # Generate a Pandas DF for the betting history created from the generate_bets method
        self.betting_history = self.generate_bets()

    def generate_bets(self):
        # Store the bet dates and amounts in a list
        bet_dates = []
        bet_amount = []
        # Generate random bets until the total spend has been reached
        while True:
            # Sample from the weeks using the Adjusted Racing probability as the distribution
            bet_date = self.racing_df.sample(1, weights=self.racing_df['Racing Prob']).index[0]
            # Use the gamma distribution with a mean of 20 to create a random bet size
            # Add the size multiplier if the bet occurs after the Bonus Bet date
            bet_size = np.random.gamma(10, 2) * self.bonusbet_sizemultiplier if bet_date >= self.random_bonusbet_date \
                else np.random.gamma(10, 2)
            bet_dates.append(bet_date)
            bet_amount.append(bet_size)
            # Check to see if the total spend has been reached and exit the loop if so
            if sum(bet_amount) >= self.total_spend:
                break
        # Return a data frame with the betting history and some customer information in case required
        return pd.DataFrame({"Customer ID": [self.customer_id] * len(bet_dates),
                             "Bet ID": range(1, len(bet_dates) + 1),
                             "Bet Date": bet_dates,
                             "Bet Amount": bet_amount})

# TODO: add this as a class method in the Customer class create a method for now
def calculate_jensenshannon(customers, remove_melbourne_cup=False):
    """
    function to return the Jensen-Shannon divergence between the bet frequency and mean bet size
    before and after the bonus bet was applied
    :param customers: dict of Customer Class
    :param remove_melbourne_cup: Bool
    :return: tuple len 2 (bet amount, bet frequency) Jensen-Shannon divergence
    """
    # Create lists to stor the mean amount and frequency
    amount = []
    freq = []
    for key, item in customers.items():
        # Remove the Melbourne Cup date as it is an outlier if remove_melbourne_cup=True
        if remove_melbourne_cup:
            df = item.betting_history[item.betting_history["Bet Date"] != pd.to_datetime("2020-11-01")]
        else:
            df = item.betting_history
        # Calculate the mean bet amount before the bonus date
        amount.append(df.loc[
                          df['Bet Date'] < item.random_bonusbet_date, "Bet Amount"].mean())
        # Calculate the mean bet amount after the bonus date
        amount.append(df.loc[
                          df['Bet Date'] >= item.random_bonusbet_date, "Bet Amount"].mean())
        # TODO: replace these try and except clauses
        try:
            # Get a weekly frequency as bets per week before bonus date
            freq.append(df.loc[df['Bet Date'] < item.random_bonusbet_date, "Bet Date"].count() /
                        df.loc[df['Bet Date'] < item.random_bonusbet_date, "Bet Date"].nunique())
        except:
            freq.append(np.NaN)
        try:
            # Get a weekly frequency as bets per week after bonus date
            freq.append(df.loc[df['Bet Date'] >= item.random_bonusbet_date, "Bet Date"].count() /
                        df.loc[df['Bet Date'] >= item.random_bonusbet_date, "Bet Date"].nunique())
        except:
            freq.append(np.NaN)
    # Put the results in a DataFrame
    linear_summary_df = pd.DataFrame({"Mean Amount": amount,
                                      "Weekly Frequency": freq,
                                      "Bonus Bet Flag": ["Before Bonus Bet", "After Bonus Bet"] * (
                                                  len(amount) // 2),
                                      "customer ID": np.repeat(list(customers.keys()), 2)})
    def _calculate_distribution(variable, bonus_bet_flag):
        """
        returns the distribution as a series of a variable using 'Bonus Bet Flag' to subset the distribution
        :param variable: str -> "Mean Amount" or "Weekly Frequency"
        :param Bonus_Bet_Flag: str -> "Before Bonus Bet" or "After Bonus Bet"
        :return: pandas Series length 200
        """
        return (linear_summary_df[linear_summary_df['Bonus Bet Flag'] == bonus_bet_flag][variable]
                .pipe(lambda s: pd.Series(np.histogram(s,
                                                       range=(int(linear_summary_df[variable].min()),
                                                              int(linear_summary_df[variable].max()) + 1),
                                                       bins=200)))
                .pipe(lambda s: pd.Series(s[0], index=s[1][:-1]))
                .pipe(lambda s: s / s.sum())
                )
    # calculate the distributions of the mean amount and bet frequency before and after a bonus bet
    amount_distribution_before = _calculate_distribution("Mean Amount", "Before Bonus Bet")
    amount_distribution_after = _calculate_distribution("Mean Amount", "After Bonus Bet")
    freq_distribution_before = _calculate_distribution("Weekly Frequency", "Before Bonus Bet")
    freq_distribution_after = _calculate_distribution("Weekly Frequency", "After Bonus Bet")

    # Return the Jensen-Shannon divergence
    return (jensenshannon(amount_distribution_before, amount_distribution_after),
            jensenshannon(freq_distribution_before, freq_distribution_after))

if __name__ == "__main__":
    # Generate 10,000 customers with the same bonus bet applied
    customers = {i: Customer(customer_id=i,
                             random_bonusbet=True) for i in range(1, 10001)}
    with open(Path().joinpath("Output", "customers10k.pkl"), 'wb') as pkl:
        pickle.dump(customers, pkl)

    # Create a dataframe to store all the customers
    total_customer_betting_df = pd.DataFrame(columns=customers[1].betting_history.columns)
    for customer_id, customer in customers.items():
        total_customer_betting_df = pd.concat([total_customer_betting_df, customer.betting_history])
    # Group by the date to check the time-series of the total bets placed
    total_yearly_series = total_customer_betting_df[["Bet Date", "Bet Amount"]].groupby(["Bet Date"]).sum()
    # Plot the weekly series
    ax = total_yearly_series.plot(legend=False, ylabel="Total Bet Amount")
    # ax.axvline(customers[1].random_bonusbet_date, ls='--')
    # plt.text(customers[1].random_bonusbet_date, float(total_yearly_series.min()), 'Bonus Bet Date')
    plt.text(pd.to_datetime('2020-11-01'), float(total_yearly_series.max()), 'Melbourne Cup')
    plt.savefig(Path().joinpath("Plots", "Total Betting Distribution Random Date.png"), bbox_inches="tight")

    # Look at the mean amount as a function of betting frequency
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
            result = calculate_jensenshannon(customers, True)
            amount_js.append(result[0])
            freq_js.append(result[1])
    # plot results
    fig, ax = plt.subplots()
    im = ax.imshow(np.resize(amount_js, (11, 11)), origin='lower')
    cbar =  ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Jensen-Shannon Divergence', rotation=-90, va='bottom')
    ax.set_xticks(np.arange(11))
    ax.set_yticks(np.arange(11))
    ax.set_xticklabels([str(x) for x in np.linspace(1,1.5,11)])
    ax.set_yticklabels([str(x) for x in np.linspace(1, 1.5, 11)])
    ax.set_xlabel('Bonus Bet Frequency Multiplier')
    ax.set_ylabel('Bonus Bet Size Multiplier')
    plt.savefig(Path().joinpath("Plots", "Amount_JS.png"))