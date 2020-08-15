import { ITweets } from "../interfaces/schemas/tweets.interfaces";
import { tweetModel } from "../models/tweet.model";

export class TweetService {
  getById = async (tweetId: string): Promise<ITweets> => {
    return await tweetModel.findOne({ tweetId }).lean();
  };

  getByUser = async (userId: string): Promise<ITweets[]> => {
    return await tweetModel.find({ userId }).lean();
  };

  create = async (data: ITweets): Promise<void> => {
    if (!(await tweetModel.exists({ tweetId: data.tweetId })))
      await tweetModel.create(data);
    console.dir("Created Tweet");
  };
}
