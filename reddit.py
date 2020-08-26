import configparser
import praw
from ketek import KetekChecker

config = configparser.ConfigParser()
config.read('config.ini')
reddit = praw.Reddit(user_agent="Ketek-Bot (by /u/MateoPeri)",
                     client_id=config['AUTH']['client_id'], client_secret=config['AUTH']['client_secret'],
                     username=config['AUTH']['username'], password=config['AUTH']['password'])

reply_text = open('reply_template.md').read()
ketek = KetekChecker(**config['KETEK_SCORER'])


def process_submission(s):
    print('\nChecking', s.permalink)
    if isinstance(s, praw.reddit.models.Submission):
        if not s.is_self:
            return
        print('Type: submission')
        txt = s.selftext
    elif isinstance(s, praw.reddit.models.Comment):
        print('Type: comment')
        txt = s.body
    else:
        print('what')
        return
    is_ketek, score = ketek.check_ketek(txt)
    if is_ketek:
        print('Is ketek!')
        print(txt)
        reply = reply_text.format(score)
        print(reply)
        # submission.reply(reply_text)


"""
Taken from: https://gist.github.com/MrEdinLaw/9d50507a037f2e2f54b76d2cadffc72a
"""
def submissions_and_comments(subreddit, **kwargs):
    results = []
    results.extend(subreddit.new(**kwargs))
    results.extend(subreddit.comments(**kwargs))
    # results.sort(key=lambda post: post.created_utc, reverse=True)
    return results


def main():
    subreddit = reddit.subreddit("KETEK+Cosmere+BrandonSanderson+Stormlight_Archive+cremposting+Mistborn")
    stream = praw.models.util.stream_generator(lambda **kwargs: submissions_and_comments(subreddit, **kwargs))
    for submission in stream:
        process_submission(submission)


if __name__ == '__main__':
    main()
