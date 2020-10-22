export interface IStreamTweet {
  id_str: string;
  created_at: string;
  geo: any;
  place: IPlaces;
  coordinates: any;
  user: IUser;
  retweeted_status?: IRetweeted;
  extended_tweet?: IExtended;
  reply_count?: number;
  retweet_count?: number;
  favorite_count?: number;
}

interface IUser {
  id_str: string;
  name: string;
  screen_name: string;
  location?: string;
  description?: string;
  verified: boolean;
  followers_count: number;
  friends_count: number;
  created_at: string;
  profile_image_url?: string;
  profile_background_image_url?: string;
  favourites_count: number;
  statuses_count: number;
  url?: string;
}

interface IExtended {
  full_text: string;
}

interface IRetweeted {
  created_at: string;
  id_str: string;
  user: IUser;
  extended_tweet: IExtended;
  reply_count: number;
  retweet_count: number;
  favorite_count: number;
}

interface IPlaces {
  id: string;
  url: string;
  place_type: string;
  name: string;
  full_name: string;
  country_code: string;
  country: string;
  bounding_box: any;
  attributes: {};
}
