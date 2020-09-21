import * as AWS from "aws-sdk";
import { awsCredentials } from "../credentials/credentials.service";

export class ComprehendService {
  private readonly comprehend: AWS.Comprehend;
  constructor() {
    this.comprehend = new AWS.Comprehend({ credentials: awsCredentials, region: 'us-east-1' });
  }

  analyzeSentiment = async (text: string, lang: string) => {
    const params = {
      LanguageCode: lang,
      Text: text,
    };
    return await this.comprehend.detectSentiment(params).promise();
  };
}
