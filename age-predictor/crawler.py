from icrawler.builtin import BaiduImageCrawler, GoogleImageCrawler

def crawl(storage_dir,age_range,age_interval,num_img_per_class):
    y = [x for x in range(age_range[0], age_range[1] + 1, age_interval)]
    if y[0] == 0: y[0] = y[0] + 1
    image_number = [num_img_per_class for i in range(4)]
    #two different crawler from Google and Baidu search engine
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4)
    baidu_crawler = BaiduImageCrawler(parser_threads=2, downloader_threads=4)
    for c in y:
        keyword_en = str(c) + " year old" # english key words
        keyword_chn = str(c) + "岁" #chinese key words
        #setting sub-dir
        storage = storage_dir + str(c)
        google_crawler.storage.root_dir = storage
        baidu_crawler.storage.root_dir = storage
        #crawling
        google_crawler.crawl(keyword=keyword_en, file_idx_offset=0, max_num=image_number[0],
                             date_min=None, date_max=None, min_size=(200, 200), max_size=None)
        google_crawler.crawl(keyword=keyword_chn, file_idx_offset=image_number[0], max_num=image_number[1],
                             date_min=None, date_max=None, min_size=(200, 200), max_size=None)
        baidu_crawler.crawl(keyword=keyword_en, file_idx_offset=sum(image_number[0:2]), max_num=image_number[2],
                            min_size=None, max_size=None)
        baidu_crawler.crawl(keyword=keyword_chn, file_idx_offset=sum(image_number[0:3]), max_num=image_number[3],
                            min_size=None, max_size=None)

if __name__ == "__main__":
    dir = "f:/blob/ds/age-predictor/" # root dir
    age_range = [0,100] # range of age
    age_interval = 5 # interval between each age class
    num_img_per_class = 50  # number of images that each crawler craws in each age range
    # return at most (age_range/age_interval+1)*4*num_img_per_class images
    crawl(dir,age_range,age_interval,num_img_per_class)