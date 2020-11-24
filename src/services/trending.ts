import { ComprehendService } from "../config/aws/comprehend/comprehend.service";
import { environment } from "../config/environment/environment";
import { IStreamTweet } from "../interfaces/tweet.interface";
import { IPoliticalResponse, ITrends } from "../interfaces/trends.interface";
import NodeGeocoder from "node-geocoder";
import { TweetService } from "./tweet";
import { UserService } from "./user";
import Twitter from "twitter-lite";
import axios from "axios";

export class TrendingService {
  // private filterRegex = new RegExp(
  //   "Poder|política|izquierda|derecha|tibio|Petro|Uribe|Duque|Farc|ELN|Disidencias|castrochavismo|cocaína|mermelada|corruptos|petristas|carroñeros|uribestias|terroristas|protesta|minga|democracia",
  //   "i"
  // );
  private twitter: Twitter;
  private userService: UserService;
  private tweetService: TweetService;
  private parameters: { track: string; locations: string; language: string };
  private comprehendService: ComprehendService;
  private geocoder: NodeGeocoder.Geocoder;
  constructor() {
    this.twitter = this.setup();
    this.tweetService = new TweetService();
    this.userService = new UserService();
    this.comprehendService = new ComprehendService();
    this.parameters = {
      track: "",
      locations: "-78.473824,1.073885,-70.124214,12.573362",
      language: "es",
    };
    this.geocoder = NodeGeocoder({
      provider: "openstreetmap",
    });
  }

  private setup = (): Twitter => {
    return new Twitter({
      consumer_key: environment.twitterKey,
      consumer_secret: environment.twitterSecret,
      access_token_key: environment.twitterToken,
      access_token_secret: environment.twitterAccessSecret,
      version: "1.1",
    });
  };

  public stream = async () => {
    try {
      const trends: ITrends[] = await this.twitter.get("trends/place", {
        id: 368148,
      });
      let track = "";
      for (let trend of trends[0].trends) {
        track += `${trend.name},`;
      }
      this.parameters.track = track.substring(0, track.length - 1);
    } catch (error) {
      console.error(error.response);
      throw new Error("Trending error");
    }
    console.dir(this.parameters);
    this.twitter
      .stream("statuses/filter", this.parameters)
      .on("start", (response) => console.log("Connected to Stream"))
      .on("data", (tweet: IStreamTweet) =>
        (async () => {
          if (tweet.place)
            if (tweet.place.country === "Colombia")
              if (tweet.extended_tweet) {
                const political = await axios.post<IPoliticalResponse>(
                  "http://54.205.200.112:8000/predict/political",
                  {
                    text: tweet.extended_tweet,
                  }
                );
                console.dir({ political: political.data.accuracy });
                if (
                  tweet.extended_tweet.full_text &&
                  political.data.accuracy > 0.65
                  // this.filterRegex.test(tweet.extended_tweet.full_text)
                )
                  //console.dir(tweet.extended_tweet.full_text);
                  try {
                    const sentiment = await this.comprehendService.analyzeSentiment(
                      tweet.extended_tweet.full_text,
                      "es"
                    );
                    const location = await this.geocoder.geocode(
                      `${tweet.place.name} Colombia`
                    );
                    console.dir({
                      mixed: sentiment.SentimentScore.Mixed,
                      negative: sentiment.SentimentScore.Negative,
                      neutral: sentiment.SentimentScore.Neutral,
                      positive: sentiment.SentimentScore.Positive,
                      predominant: sentiment.Sentiment,
                    });
                    await this.tweetService.create({
                      userId: tweet.user.id_str,
                      tweetId: tweet.id_str,
                      createdAt: tweet.created_at,
                      favorites: tweet.favorite_count,
                      text: tweet.extended_tweet.full_text,
                      replies: tweet.reply_count,
                      retweets: tweet.retweet_count,
                      hashtag: this.parameters.track,
                      location: {
                        latitude: location[0].latitude,
                        longitude: location[0].longitude,
                        city: tweet.place.name
                      },
                      accuracy: {
                        political: political.data.accuracy,
                      },
                      sentimentScore: {
                        mixed: sentiment.SentimentScore.Mixed,
                        negative: sentiment.SentimentScore.Negative,
                        neutral: sentiment.SentimentScore.Neutral,
                        positive: sentiment.SentimentScore.Positive,
                        predominant: sentiment.Sentiment,
                      },
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
                    if (tweet.place)
                      if (tweet.place.country === "Colombia") {
                        const political = await axios.post<IPoliticalResponse>(
                          "http://54.205.200.112:8000/predict/political",
                          {
                            text: tweet.retweeted_status.extended_tweet,
                          }
                        );
                        if (
                          tweet.retweeted_status.extended_tweet.full_text &&
                          political.data.accuracy > 0.65
                          // this.filterRegex.test(
                          //   tweet.retweeted_status.extended_tweet.full_text
                          // )
                        ) {
                          // console.dir(
                          //   tweet.retweeted_status.extended_tweet.full_text
                          // );
                          try {
                            const sentiment = await this.comprehendService.analyzeSentiment(
                              tweet.retweeted_status.extended_tweet.full_text,
                              "es"
                            );
                            const location = await this.geocoder.geocode(
                              `${tweet.place.name} Colombia`
                            );
                            console.dir({
                              mixed: sentiment.SentimentScore.Mixed,
                              negative: sentiment.SentimentScore.Negative,
                              neutral: sentiment.SentimentScore.Neutral,
                              positive: sentiment.SentimentScore.Positive,
                              predominant: sentiment.Sentiment,
                            });
                            await this.tweetService.create({
                              userId: tweet.retweeted_status.user.id_str,
                              tweetId: tweet.retweeted_status.id_str,
                              createdAt: tweet.retweeted_status.created_at,
                              favorites: tweet.retweeted_status.favorite_count,
                              text:
                                tweet.retweeted_status.extended_tweet.full_text,
                              replies: tweet.retweeted_status.reply_count,
                              retweets: tweet.retweeted_status.retweet_count,
                              hashtag: this.parameters.track,
                              location: {
                                latitude: location[0].latitude,
                                longitude: location[0].longitude,
                                city: tweet.place.name
                              },
                              accuracy: {
                                political: political.data.accuracy,
                              },
                              sentimentScore: {
                                mixed: sentiment.SentimentScore.Mixed,
                                negative: sentiment.SentimentScore.Negative,
                                neutral: sentiment.SentimentScore.Neutral,
                                positive: sentiment.SentimentScore.Positive,
                                predominant: sentiment.Sentiment,
                              },
                            });
                            await this.userService.create({
                              url: tweet.retweeted_status.user.url,
                              createdAt: tweet.retweeted_status.user.created_at,
                              favourites:
                                tweet.retweeted_status.user.favourites_count,
                              followers:
                                tweet.retweeted_status.user.followers_count,
                              friends:
                                tweet.retweeted_status.user.friends_count,
                              name: tweet.retweeted_status.user.name,
                              statuses:
                                tweet.retweeted_status.user.statuses_count,
                              userId: tweet.retweeted_status.user.id_str,
                              username: tweet.retweeted_status.user.screen_name,
                              verified: tweet.retweeted_status.user.verified,
                              backgroundUrl:
                                tweet.retweeted_status.user
                                  .profile_background_image_url,
                              description:
                                tweet.retweeted_status.user.description,
                              location: tweet.retweeted_status.user.location,
                              profileUrl:
                                tweet.retweeted_status.user.profile_image_url,
                            });
                          } catch (error) {
                            console.dir(error);
                          }
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
