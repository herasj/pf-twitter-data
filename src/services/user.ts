import { IUser } from "../interfaces/schemas/users.interfaces";
import { userModel } from "../models/user.model";

export class UserService {
  getById = async (tweetId: string): Promise<IUser> => {
    return await userModel.findOne({ tweetId }).lean();
  };

  getByUser = async (userId: string): Promise<IUser[]> => {
    return await userModel.find({ userId }).lean();
  };

  create = async (data: IUser): Promise<void> => {
    if (!(await userModel.exists({ userId: data.userId }))) {
      await userModel.create(data);
    }
    const user = await userModel.findOne({ userId: data.userId }).lean();
    await userModel.findOneAndUpdate(
      { userId: data.userId },
      { counter: user.counter + 1 }
    );
    console.dir("Created User");
  };
}
