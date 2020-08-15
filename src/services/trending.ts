import Twitter from "twitter-lite";
import { environment } from "../config/environment/environment";
import { IStreamTweet } from "../interfaces/tweet.interface";
import { TweetService } from "./tweet";
import { UserService } from "./user";
import moment from "moment";

export class TrendingService {
  private twitter: Twitter;
  private tweetService: TweetService;
  private userService: UserService;
  private parameters: { track: string };

  constructor() {
    this.twitter = this.setup();
    this.tweetService = new TweetService();
    this.userService = new UserService();
    this.parameters = {
      track: "Duque",
    };
  }

  private setup = (): Twitter => {
    return new Twitter({
      consumer_key: environment.twitterKey,
      consumer_secret: environment.twitterSecret,
      access_token_key: environment.twitterToken,
      access_token_secret: environment.twitterAccessSecret,
    });
  };

  public stream = () => {
    this.twitter
      .stream("statuses/filter", this.parameters)
      .on("start", (response) => console.log("Connected to Stream"))
      .on("data", (tweet: IStreamTweet) =>
        (async () => {
          if (tweet.extended_tweet) {
            if (tweet.extended_tweet.full_text)
              try {
                await this.tweetService.create({
                  userId: tweet.user.id_str,
                  tweetId: tweet.id_str,
                  createdAt: tweet.created_at,
                  favorites: tweet.favorite_count,
                  text: tweet.extended_tweet.full_text,
                  replies: tweet.reply_count,
                  retweets: tweet.retweet_count,
                  hashtag: this.parameters.track,
                });
                await this.userService.create({
                  url: tweet.user.url,
                  createdAt: tweet.user.created_at,
                  favourites: tweet.user.favourites_count,
                  followers: tweet.user.followers_count,
                  friends: tweet.user.friends_count,
                  name: tweet.user.name,
                  statuses: tweet.user.statuses_count,
                  userId: tweet.user.id_str,
                  username: tweet.user.screen_name,
                  verified: tweet.user.verified,
                  backgroundUrl: tweet.user.profile_background_image_url,
                  description: tweet.user.description,
                  location: tweet.user.location,
                  profileUrl: tweet.user.profile_image_url,
                });
              } catch (error) {
                console.dir(error);
              }
          } else {
            if (tweet.retweeted_status)
              if (tweet.retweeted_status.extended_tweet)
                if (tweet.retweeted_status.extended_tweet.full_text) {
                  try {
                    await this.tweetService.create({
                      userId: tweet.retweeted_status.user.id_str,
                      tweetId: tweet.retweeted_status.id_str,
                      createdAt: tweet.retweeted_status.created_at,
                      favorites: tweet.retweeted_status.favorite_count,
                      text: tweet.retweeted_status.extended_tweet.full_text,
                      replies: tweet.retweeted_status.reply_count,
                      retweets: tweet.retweeted_status.retweet_count,
                      hashtag: this.parameters.track,
                    });
                    await this.userService.create({
                      url: tweet.retweeted_status.user.url,
                      createdAt: tweet.retweeted_status.user.created_at,
                      favourites: tweet.retweeted_status.user.favourites_count,
                      followers: tweet.retweeted_status.user.followers_count,
                      friends: tweet.retweeted_status.user.friends_count,
                      name: tweet.retweeted_status.user.name,
                      statuses: tweet.retweeted_status.user.statuses_count,
                      userId: tweet.retweeted_status.user.id_str,
                      username: tweet.retweeted_status.user.screen_name,
                      verified: tweet.retweeted_status.user.verified,
                      backgroundUrl:
                        tweet.retweeted_status.user
                          .profile_background_image_url,
                      description: tweet.retweeted_status.user.description,
                      location: tweet.retweeted_status.user.location,
                      profileUrl: tweet.retweeted_status.user.profile_image_url,
                    });
                  } catch (error) {
                    console.dir(error);
                  }
                }
          }
        })()
      )
      .on("ping", () => console.log("ping"))
      .on("error", (error) => console.log("error", error))
      .on("end", (response) => console.log("end"));
  };
}
