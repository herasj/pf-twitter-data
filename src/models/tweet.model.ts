import { Schema, model } from "mongoose";
import { ITweets, ITweetsModel } from "../interfaces/schemas/tweets.interfaces";

const TweetSchema = new Schema<ITweets>(
  {
    tweetId: { type: String, required: true },
    createdAt: { type: String, required: true },
    userId: { type: String, required: true },
    text: { type: String, required: true },
    replies: { type: Number, required: true },
    retweets: { type: Number, required: true },
    favorites: { type: Number, required: true },
    hashtag: { type: String, required: true },
  },
  { id: false, versionKey: false }
);

export const tweetModel = model<ITweetsModel>("tweets", TweetSchema);
