import { Schema, model } from "mongoose";
import {
  ITweets,
  ILocation,
  ISentiment,
  ITweetsModel,
} from "../interfaces/schemas/tweets.interfaces";

const SentimentSchema = new Schema<ISentiment>(
  {
    predominant: { type: String, required: true },
    positive: { type: Number, required: true },
    negative: { type: Number, required: true },
    neutral: { type: Number, required: true },
    mixed: { type: Number, required: true },
  },
  {
    _id: false,
    versionKey: false,
    id: false,
    timestamps: true,
  }
);

const LocationSchema = new Schema<ILocation>(
  {
    longitude: { type: Number, required: true },
    latitude: { type: Number, required: true },
  },
  {
    _id: false,
    versionKey: false,
  }
);

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
    url: { type: String, required: false },
    location: LocationSchema,
    sentimentScore: SentimentSchema,
  },
  { id: false, versionKey: false }
);

export const tweetModel = model<ITweetsModel>("tweets", TweetSchema);
