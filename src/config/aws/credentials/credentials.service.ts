import * as AWS from 'aws-sdk'
import { environment } from '../../environment/environment'

export const awsCredentials = new AWS.Credentials({
  accessKeyId: environment.awsAccess,
  secretAccessKey: environment.awsSecret,
});