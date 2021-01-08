# import standard data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import jensenshannon
# import Path to read the csv file in an OS independent way
from pathlib import Path


# Create a customer class to
class Customer(object):
    """
    class to create a customer to generate synthetic betting behaviour
    Parameters
    ----------
    customer_id : int
    user defined numeric customer id

    bonusbet_sizemultier : float -> defult=1.2
    size multiplier for the change in betting history after a bonus bet

    bonusbet_freqmultiplier : float -> defult=1.2
    frequency multiplier for the change in betting history after a bonus bet

    random_bonusbet : bool -> defult=False
    Random bonus bet for each customer if True

    gamma_param : tuple dim 2 -> (5,150)
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
        # Which I'd assume the customer behaviour would be in practice
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
        # Change the frequency distribution by adding a multiplier
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

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


if __name__ == "__main__":
    # Generate 10,000 customers with the same bonus bet applied
    customers = {i: Customer(i) for i in range(1, 10001)}
    # Create a dataframe to store all the customers
    total_customer_betting_df = pd.DataFrame(columns=customers[1].betting_history.columns)
    for key, item in customers.items():
        total_customer_betting_df = pd.concat([total_customer_betting_df, item.betting_history])

    # Group by the date to check the time-series of the total bets placed
    total_yearly_series = total_customer_betting_df[["Bet Date", "Bet Amount"]].groupby(["Bet Date"]).sum()
    # Plot the weekly series
    ax = total_yearly_series.plot(legend=False, ylabel="Total Bet Amount")
    ax.axvline(customers[1].random_bonusbet_date, ls='--')
    plt.text(customers[1].random_bonusbet_date, float(total_yearly_series.min()), 'Bonus Bet Date')
    plt.text(pd.to_datetime('2020-11-01'), float(total_yearly_series.max()), 'Melbourne Cup')
    plt.savefig("Total Betting Distribution.png", bbox_inches="tight")
    # Look at the mean amount as a function of betting frequency
    # TODO: add this as a class method in the Customer class create a method for now
    def fit_linear_model(customers, remove_melbourne_cup=False):
        amount = []
        freq = []
        for key, item in customers.items():
            if remove_melbourne_cup:
                df = item.betting_history[item.betting_history["Bet Date"] != pd.to_datetime("2020-11-01")]
            else:
                df = item.betting_history
            amount.append(df.loc[
                df['Bet Date'] < item.random_bonusbet_date, "Bet Amount"].mean())
            amount.append(df.loc[
                df['Bet Date'] >= item.random_bonusbet_date, "Bet Amount"].mean())
            try:
                freq.append(7 * df.loc[df['Bet Date'] < item.random_bonusbet_date, "Bet Date"].count() / \
                            df.loc[df['Bet Date'] < item.random_bonusbet_date, "Bet Date"].nunique())
            except:
                freq.append(np.NaN)
            try:
                freq.append(7 * df.loc[df['Bet Date'] >= item.random_bonusbet_date, "Bet Date"].count() / \
                            df.loc[df['Bet Date'] >= item.random_bonusbet_date, "Bet Date"].nunique())
            except:
                freq.append(np.NaN)

        linear_summary_df = pd.DataFrame({"Mean Amount": amount,
                                          "Weekly Frequency": freq,
                                          "Bonus Bet Flag": ["Before Bonus Bet", "After Bonus Bet"]*(len(amount)//2),
                                          "customer ID": np.repeat(list(customers.keys()), 2)})

        def calculate_distribution(Variable, Bonus_Bet_Flag):
            return (linear_summary_df[linear_summary_df['Bonus Bet Flag'] == Bonus_Bet_Flag][Variable]
                 .pipe(lambda s: pd.Series(np.histogram(s,
                                                        range=(int(linear_summary_df[Variable].min()),
                                                               int(linear_summary_df[Variable].max()) + 1),
                                                        bins=200)))
                 .pipe(lambda s: pd.Series(s[0], index=s[1][:-1]))
                 .pipe(lambda s: s / s.sum())
                 )

        amount_distribution_before = calculate_distribution("Mean Amount", "Before Bonus Bet")
        amount_distribution_after = calculate_distribution("Mean Amount", "Before Bonus Bet")
        freq_distribution_before = calculate_distribution("Mean Amount", "Before Bonus Bet")
        freq_distribution_after = calculate_distribution("Mean Amount", "Before Bonus Bet")

        return (jensenshannon(amount_distribution_before,amount_distribution_after),
                jensenshannon(freq_distribution_before,freq_distribution_after))


        # remove_nan_inf_bool = (linear_summary_df.notna().all(axis=1)) & \
        #                       ~(np.isinf(linear_summary_df[["Weekly Frequency", "Mean Amount"]]).any(axis=1))
        #
        # before_bonus_bet_reg = LinearRegression().fit(np.array(
        #     linear_summary_df[(remove_nan_inf_bool) & (linear_summary_df["Bonus Bet Flag"] == "Before Bonus Bet")]["Weekly Frequency"]).reshape(-1,1),
        #     linear_summary_df[(remove_nan_inf_bool) & (linear_summary_df["Bonus Bet Flag"] == "Before Bonus Bet")]["Mean Amount"])
        # after_bonus_bet_reg = LinearRegression().fit(np.array(
        #     linear_summary_df[(remove_nan_inf_bool) & (linear_summary_df["Bonus Bet Flag"] == "After Bonus Bet")]["Weekly Frequency"]).reshape(-1,1),
        #     linear_summary_df[(remove_nan_inf_bool) & (linear_summary_df["Bonus Bet Flag"] == "After Bonus Bet")]["Mean Amount"])
        # return before_bonus_bet_reg.coef_, after_bonus_bet_reg.coef_

    results = []
    for bonus_bet_size_multiplier in [1, 1.01, 1.05, 1.1, 1.15]:
        for bonus_bet_freq_multiplier in [1, 1.01, 1.05, 1.1, 1.15]:
            customers = {i: Customer(i,
                                     bonusbet_sizemultiplier=bonus_bet_size_multiplier,
                                     bonusbet_freqmultiplier=bonus_bet_freq_multiplier,
                                     random_bonusbet=True
                                     ) for i in range(1, 1001)}
            results.append(fit_linear_model(customers, True))

    int(linear_summary_df["Mean Amount"].min())
