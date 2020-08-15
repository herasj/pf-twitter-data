import { Document } from "mongoose";
export interface IUserModel extends Document {
  userId: string;
  name: string;
  username: string;
  location?: string;
  description?: string;
  verified: boolean;
  followers: number;
  friends: number;
  createdAt: string;
  profileUrl?: string;
  backgroundUrl?: string;
  favourites: number;
  statuses: number;
  counter?: number;
  url?: string;
}

export interface IUser {
  userId: string;
  name: string;
  username: string;
  location?: string;
  description?: string;
  verified: boolean;
  followers: number;
  friends: number;
  createdAt: string;
  profileUrl?: string;
  backgroundUrl?: string;
  favourites: number;
  statuses: number;
  counter?: number;
  url?: string;
}
