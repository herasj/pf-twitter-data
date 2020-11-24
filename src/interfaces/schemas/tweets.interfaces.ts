import { Document, Types } from "mongoose";
export interface ITweetsModel extends Document {
  tweetId: string;
  userId: string;
  createdAt: string;
  text: string;
  replies: number;
  retweets: number;
  favorites: number;
  hashtag: string;
  location: ILocation;
  accuracy: {
    political?: number;
    hate?: number;
  };
  url?: string;
  sentimentScore: ISentiment;
}

export interface ISentiment {
  predominant: string;
  positive: number;
  negative: number;
  neutral: number;
  mixed: number;
}

export interface ITweets {
  tweetId: string;
  userId: string;
  createdAt: string;
  text: string;
  replies: number;
  retweets: number;
  favorites: number;
  hashtag: string;
  location: ILocation;
  accuracy: {
    political?: number;
    hate?: number;
  };
  url?: string;
  sentimentScore: ISentiment;
}

export interface ILocation {
  latitude: number;
  longitude: number;
  city: string;
}
