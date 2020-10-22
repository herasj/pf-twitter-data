interface ITrendInfo {
  name: string;
  url: string;
  promoted_content: any;
  query: string;
  tweet_volume: number;
}

interface ITrendLocation {
  name: string;
  woeid: number;
}

export interface ITrends {
  trends: ITrendInfo[];
  as_of: string;
  created_at: string;
  locations: ITrendLocation[];
}
