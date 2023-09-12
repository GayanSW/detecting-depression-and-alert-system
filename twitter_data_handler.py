from datetime import datetime, timedelta
import GetOldTweets3 as got
from constants import TWITTER_USERNAME


def get_tweets():
    days = 14
    date_n_days_ago = datetime.today() - timedelta(days=days)
    until_date = datetime.today() + timedelta(days=1)
    tweet_criteria = got.manager.TweetCriteria().setUsername(TWITTER_USERNAME).setSince(date_n_days_ago.date().strftime('%Y-%m-%d')).setUntil(until_date.date().strftime('%Y-%m-%d'))
    return got.manager.TweetManager.getTweets(tweet_criteria)
