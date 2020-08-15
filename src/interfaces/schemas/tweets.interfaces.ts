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
}
