from src.data.crawler.TweetsCrawler import TweetsCrawler
import src.utils as utils
from src.constants import RAW_DATA_URL
import sys
def gather_raw_data(count_per_class, raw_data_output_dir=RAW_DATA_URL):
    crawler = TweetsCrawler()
    class_ids = crawler.classesData.get_total_class_ids()
    for class_id in class_ids:
        class_name = crawler.classesData.get_class_name(class_id)
        print(f"Start getting data for class {class_name}.")
        tweets_id = crawler.gather_tweets_id(class_id, count_per_class)
        tweets_text = crawler.gather_tweets_text(tweets_id, class_id)
        print(f"Gathered total of {len(tweets_text)} for class {class_name}.")
        url = f"{raw_data_output_dir}/{class_name}_raw_text.tsv"
        utils.write_lines(url, tweets_text, "\t")
        print(f"Saved in {url}.")
        utils.write_lines(f"{raw_data_output_dir}/{class_name}_raw_ids.tsv", tweets_id, "\t")

if __name__ == "__main__":
    count_per_class = int(sys.argv[1])
    try:
        url = sys.argv[2]
    except:
        url = RAW_DATA_URL
    gather_raw_data(count_per_class, url)
