import { IEnvironment } from "../../interfaces/config/environment.interface";
require("dotenv").config(); 

export const environment: IEnvironment = {
  mongoUri: process.env.MONGO_URI,
  twitterKey: process.env.TWITTER_API_KEY,
  twitterSecret: process.env.TWITTER_API_SECRET,
  twitterToken: process.env.TWITTER_ACCESS_TOKEN,
  twitterAccessSecret: process.env.TWITTER_ACCESS_SECRET,
  awsAccess: process.env.AWS_ACCESS,
  awsSecret: process.env.AWS_SECRET
};
